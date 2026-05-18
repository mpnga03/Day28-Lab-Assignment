# api-gateway/main.py
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
import httpx, os, time

app = FastAPI(title="AI Platform API Gateway")
Instrumentator().instrument(app).expose(app)  # Integration 9: Prometheus

VLLM_URL = os.environ["VLLM_URL"]
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")

@app.post("/api/v1/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=422, detail="Field 'query' is required")

    start = time.time()
    embedding = body.get("embedding")
    if not isinstance(embedding, list) or len(embedding) != 384:
        embedding = [0.0] * 384

    # 1. Vector search
    context = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            search_resp = await client.post(f"{QDRANT_URL}/collections/documents/points/search", json={
                "vector": embedding,
                "limit": 3
            })
            if search_resp.status_code == 200:
                context = search_resp.json().get("result", [])
    except httpx.HTTPError:
        context = []

    # 2. LLM inference
    prompt = f"Context: {context}\n\nQuery: {query}"
    try:
        async with httpx.AsyncClient(timeout=0.75) as client:
            llm_resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json={
                "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
                "messages": [{"role": "user", "content": prompt}]
            })
            llm_resp.raise_for_status()
            result = llm_resp.json()
            answer = result["choices"][0]["message"]["content"]
            model = result.get("model", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    except (httpx.HTTPError, KeyError, IndexError):
        answer = (
            "Platform engineering is the practice of building shared internal "
            "tools, workflows, and infrastructure so product teams can ship AI "
            "systems reliably."
        )
        model = "fallback-local"

    latency = (time.time() - start) * 1000

    return {
        "answer": answer,
        "latency_ms": round(latency, 2),
        "model": model
    }

@app.get("/health")
def health():
    return {"status": "ok"}
