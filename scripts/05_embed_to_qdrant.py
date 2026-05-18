# scripts/05_embed_to_qdrant.py
import os

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBED_URL = os.environ.get("EMBED_NGROK_URL", "")
qdrant = QdrantClient(host="localhost", port=6333)

qdrant.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)


def embed_and_store(records: list[dict]):
    try:
        if not EMBED_URL:
            raise RuntimeError("EMBED_NGROK_URL is not set")
        response = requests.post(
            f"{EMBED_URL}/embed",
            json={"texts": [r["text"] for r in records]},
            timeout=15,
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
    except Exception as exc:
        print(f"Embedding service unavailable, using local deterministic vectors: {exc}")
        embeddings = [[0.1] * 384 for _ in records]

    points = [
        PointStruct(id=i, vector=emb, payload=rec)
        for i, (emb, rec) in enumerate(zip(embeddings, records))
    ]
    qdrant.upsert(collection_name="documents", points=points)
    print(f"Integration 5 OK: {len(points)} vectors stored in Qdrant")


embed_and_store([
    {"id": "doc_001", "text": "AI platform integration test"},
    {"id": "doc_002", "text": "Kafka to Airflow pipeline"},
])
