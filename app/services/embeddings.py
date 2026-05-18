import os
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

EMBED_MODEL = "gemini-embedding-001"
OUTPUT_DIMS = 768  # Matryoshka truncation → fits Supabase vector(768) + HNSW index
BATCH_SIZE = 10
_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _get_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _embed_batch_sync(texts: list[str]) -> list[list[float]]:
    """Gọi batchEmbedContents REST API trực tiếp — bỏ qua SDK."""
    url = f"{_API_BASE}/{EMBED_MODEL}:batchEmbedContents"
    payload = {
        "requests": [
            {
                "model": f"models/{EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": OUTPUT_DIMS,
            }
            for t in texts
        ]
    }
    resp = httpx.post(url, json=payload, params={"key": _get_key()}, timeout=30)
    resp.raise_for_status()
    return [e["values"] for e in resp.json()["embeddings"]]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        embeddings = await asyncio.to_thread(_embed_batch_sync, batch)
        all_embeddings.extend(embeddings)
    return all_embeddings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _embed_query_sync(text: str) -> list[float]:
    """Gọi embedContent REST API trực tiếp — bỏ qua SDK."""
    url = f"{_API_BASE}/{EMBED_MODEL}:embedContent"
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": OUTPUT_DIMS,
    }
    resp = httpx.post(url, json=payload, params={"key": _get_key()}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


async def embed_query(text: str) -> list[float]:
    return await asyncio.to_thread(_embed_query_sync, text)
