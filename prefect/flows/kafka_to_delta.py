# prefect/flows/kafka_to_delta.py
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[2] / ".prefect"))
os.environ.setdefault("PREFECT_API_URL", "http://localhost:4200/api")

import pandas as pd
from kafka import KafkaConsumer
from prefect import flow, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DELTA_LAKE_PATH = Path(os.environ.get("DELTA_LAKE_PATH", PROJECT_ROOT / "delta-lake" / "raw"))


@task
def consume_and_process():
    """Consume data from Kafka topic."""
    consumer = KafkaConsumer(
        "data.raw",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000,
        value_deserializer=lambda m: json.loads(m.decode()),
    )
    records = []
    for msg in consumer:
        records.append(msg.value)

    print(f"Consumed {len(records)} records from Kafka")
    return records


@task
def save_to_delta(records):
    """Save records to Delta Lake using parquet files."""
    if not records:
        print("No records to save")
        return

    df = pd.DataFrame(records)
    DELTA_LAKE_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DELTA_LAKE_PATH / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    print(f"Saved {len(df)} records to Delta Lake")


@flow(name="Kafka to Delta Pipeline")
def kafka_to_delta_flow():
    """Main flow: consume from Kafka and save to Delta Lake."""
    records = consume_and_process()
    save_to_delta(records)


if __name__ == "__main__":
    kafka_to_delta_flow()
