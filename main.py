import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.routers import documents, chat

app = FastAPI(
    title="TDMU SmartDoc API",
    version="1.0.0",
    description="RAG-powered document Q&A backend for TDMU SmartDoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # Bearer token không cần credentials CORS
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=["*"],
    max_age=600,
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/ping")
def ping():
    return {"status": "ok", "service": "TDMU SmartDoc API"}


@app.get("/health")
async def health():
    results = {}

    # Kiểm tra Gemini embedding với gemini-embedding-001
    try:
        import httpx as _httpx
        _key = os.getenv("GEMINI_API_KEY", "")
        _url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        _payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": "test"}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": 768,
        }
        _resp = _httpx.post(_url, json=_payload, params={"key": _key}, timeout=10)
        _vals = _resp.json().get("embedding", {}).get("values", []) if _resp.status_code == 200 else []
        results["gemini"] = f"ok (dims={len(_vals)})" if _vals else f"embed failed {_resp.status_code}: {_resp.text[:150]}"
    except Exception as e:
        results["gemini"] = f"error: {str(e)[:300]}"

    # Kiểm tra Supabase
    try:
        from app.services.database import get_supabase
        sb = get_supabase()
        sb.table("documents").select("id").limit(1).execute()
        results["supabase"] = "ok"
    except Exception as e:
        results["supabase"] = f"error: {str(e)[:120]}"

    # Kiểm tra Firebase
    try:
        import firebase_admin
        results["firebase"] = "ok" if firebase_admin._apps else "not_initialized"
    except Exception as e:
        results["firebase"] = f"error: {str(e)[:120]}"

    results["env"] = {
        "GEMINI_API_KEY": "set" if os.getenv("GEMINI_API_KEY") else "MISSING",
        "SUPABASE_URL": "set" if os.getenv("SUPABASE_URL") else "MISSING",
        "SUPABASE_SERVICE_KEY": "set" if os.getenv("SUPABASE_SERVICE_KEY") else "MISSING",
        "FIREBASE_SERVICE_ACCOUNT_JSON": "set" if os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") else "MISSING",
    }

    return results
