import os
import joblib
import pandas as pd
import json
import logging
from datetime import datetime, timezone

from quixstreams import Application
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WriteOptions


# Load environment variables (useful when working locally)
load_dotenv(".env")

# Setup logging level and format
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load configuration from environment variables
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "event-frames-model")
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "fan-speed-prediction")
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "/app/fan_speed_model.pkl")

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

# Initialize InfluxDB client
try:
    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    influx_writer = influx_client.write_api(write_options=WriteOptions(batch_size=1, flush_interval=1))
    logging.info("✅ InfluxDB client initialized successfully")
except Exception as e:
    logging.error(f"❌ Failed to initialize InfluxDB client: {e}")
    exit(1)

# Load pretrained scikit-learn model
try:
    model = joblib.load(MODEL_LOCATION)
    logging.info(f"✅ Loaded model successfully from {MODEL_LOCATION}")
except Exception as e:
    logging.error(f"❌ Failed to load model from {MODEL_LOCATION}: {e}")
    exit(1)

# Setup QuixStreams application and topics
try:
    app = Application(
        broker_address=KAFKA_BROKER,
        loglevel="INFO",
        auto_offset_reset="earliest",
        state_dir=os.path.dirname(os.path.abspath(__file__)) + "/state/",
        consumer_group="predict-from-kafka-batch"
    )
    input_topic = app.topic(KAFKA_INPUT_TOPIC, value_deserializer="json")
    producer = app.get_producer()
    output_topic = app.topic(KAFKA_OUTPUT_TOPIC, value_serializer="json")
    logging.info(f"✅ Kafka topics configured: input='{KAFKA_INPUT_TOPIC}', output='{KAFKA_OUTPUT_TOPIC}'")
except Exception as e:
    logging.error(f"❌ Failed to setup QuixStreams application or topics: {e}")
    exit(1)

def handle_message(row):
    try:
        sensor_name = row.get("name", "")
        payload = row.get("payload", {})

        features_df = pd.DataFrame([{
            "temperature": payload["temperature"],
            "humidity": payload["humidity"],
            "pressure": payload["pressure"]
        }])
        prediction = model.predict(features_df)[0]

        timestamp = datetime.fromtimestamp(
            payload.get("timestamp", datetime.now().timestamp() * 1000) / 1000.0,
            tz=timezone.utc
        )

        if sensor_name == "iot_sensor_0":
            actual_fan_speed = payload.get("fan_speed", None)
            if actual_fan_speed is not None:
                logging.info(f"[Model Sensor] {sensor_name} Actual fan_speed: {actual_fan_speed}, Predicted: {prediction}")

                # You can calculate and log MAE or other metrics here if you want
            else:
                logging.warning("[Model Sensor] fan_speed not found in payload")

            # Prepare result with both actual and predicted fan_speed
            result_payload = {
                "temperature": payload["temperature"],
                "humidity": payload["humidity"],
                "pressure": payload["pressure"],
                "fan_speed": int(actual_fan_speed) if actual_fan_speed is not None else None,
                "fan_speed_predicted": int(prediction),
                "timestamp": int(timestamp.timestamp() * 1000),
                "date": timestamp.isoformat()
            }

        else:
            # For other sensors, send only predicted fan_speed
            result_payload = {
                "temperature": payload["temperature"],
                "humidity": payload["humidity"],
                "pressure": payload["pressure"],
                "fan_speed_predicted": int(prediction),
                "timestamp": int(timestamp.timestamp() * 1000),
                "date": timestamp.isoformat()
            }

        result = {
            "id": row.get("id", ""),
            "name": sensor_name,
            "place_id": row.get("place_id", ""),
            "payload": result_payload
        }

        new_payload = json.dumps(result).encode('utf-8')

        producer.produce(
            topic=output_topic.name,
            key=sensor_name,
            value=new_payload,
            timestamp=int(timestamp.timestamp() * 1000)
        )
        logging.info(f"[📤] Sent prediction to Kafka topic: {KAFKA_OUTPUT_TOPIC}, payload: {json.dumps(result)}")


        # Write to InfluxDB (only fan_speed_predicted)
        point = (
            Point("fan_speed_prediction")
            .tag("sensor_id", row.get("id", "unknown"))
            .tag("place_id", row.get("place_id", "unknown"))
            .tag("name", sensor_name)
            .field("fan_speed_predicted", int(prediction))
            .time(timestamp)
        )
        influx_writer.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        logging.info("[📊] Wrote prediction to InfluxDB")

    except KeyError as e:
        logging.error(f"❌ Missing key in payload: {e}")
    except Exception as e:
        logging.error(f"❌ Error processing message: {e}")



# Run the streaming application
try:
    sdf = app.dataframe(input_topic)
    sdf = sdf.apply(handle_message)

    logging.info(f"🚀 Listening to Kafka topic: {KAFKA_INPUT_TOPIC}")
    app.run(sdf)
except Exception as e:
    logging.error(f"❌ Error running stream application: {e}")
    exit(1)
