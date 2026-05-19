import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from google import genai

from app.auth import get_current_user
from app.services.database import get_supabase
from app.services.embeddings import embed_query
from app.services.cache import get_cached, set_cached

router = APIRouter()

TOP_K = 3
MAX_CONTEXT_CHARS = 3000

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
GENERATE_MODEL = "gemini-2.5-flash"

MOCK_QA = [
    {
        "answer": "Dựa trên tài liệu, AI đang áp dụng kiến trúc đa phương thức (multi-modal) để cải thiện hiểu biết ngữ cảnh.",
        "citations": [{"label": "Trang", "value": "12"}, {"label": "Trang", "value": "20"}],
    },
    {
        "answer": "Học máy (Machine Learning) là nhánh của AI cho phép máy tính học từ dữ liệu mà không cần lập trình tường minh.",
        "citations": [{"label": "Trang", "value": "3"}],
    },
]
_mock_idx = 0


class AskRequest(BaseModel):
    message: str
    doc_id: str | None = None


@router.post("/ask")
async def ask(req: AskRequest, user: dict = Depends(get_current_user)):
    question = req.message.strip()
    if not question:
        raise HTTPException(400, "Câu hỏi không được để trống")

    doc_id = req.doc_id or "all"
    uid = user["uid"]

    # 1. Kiểm tra cache
    cached = get_cached(question, doc_id)
    if cached:
        return cached

    # 2. Embed câu hỏi
    try:
        q_embedding = await embed_query(question)
    except Exception as e:
        print(f"[ask] Embed error: {e}")
        return _mock_response()

    # 3. Tìm chunks qua pgvector RPC
    sb = get_supabase()
    try:
        rpc_params: dict = {
            "query_embedding": q_embedding,
            "match_count": TOP_K,
            "user_id_filter": uid,
        }
        if doc_id != "all":
            rpc_params["doc_id_filter"] = doc_id

        rpc_name = "match_chunks" if doc_id == "all" else "match_chunks_by_doc"
        res = await asyncio.to_thread(
            lambda: sb.rpc(rpc_name, rpc_params).execute()
        )
        chunks = res.data or []
    except Exception as e:
        print(f"[ask] Supabase error: {e}")
        return _mock_response()

    if not chunks:
        return _mock_response()

    # 4. Xây context + citations
    context = "\n---\n".join(
        c.get("content", "") for c in chunks
    )[:MAX_CONTEXT_CHARS]

    citations = [
        {"label": "Trang", "value": str(c.get("page_num", "?"))}
        for c in chunks
    ]

    # 5. Gọi Gemini 2.5 Flash
    prompt = (
        "Bạn là trợ lý học tập TDMU SmartDoc. "
        "Trả lời bằng tiếng Việt, ngắn gọn và chính xác.\n\n"
        f"NGỮ CẢNH TÀI LIỆU:\n{context}\n\n"
        f"CÂU HỎI: {question}\n\n"
        "TRẢ LỜI:"
    )

    try:
        response = await asyncio.to_thread(
            lambda: _client.models.generate_content(
                model=GENERATE_MODEL,
                contents=prompt,
            )
        )
        answer = response.text.strip()
    except Exception as e:
        print(f"[ask] Gemini error: {e}")
        return _mock_response()

    result = {"answer": answer, "citations": citations}
    set_cached(question, doc_id, result)
    return result


def _mock_response() -> dict:
    global _mock_idx
    mock = MOCK_QA[_mock_idx % len(MOCK_QA)]
    _mock_idx += 1
    return mock
