import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google import genai

from app.auth import get_current_user
from app.services.database import get_supabase
from app.services.embeddings import embed_query
from app.services.cache import get_cached, set_cached

router = APIRouter()

TOP_K = 5                     # tăng từ 3 lên 5 để có nhiều context hơn
MAX_CONTEXT_CHARS = 4000      # tăng từ 3000

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
GENERATE_MODEL = "gemini-2.5-flash"

MOCK_QA = [
    {
        "answer": "Dựa trên tài liệu, AI đang áp dụng kiến trúc đa phương thức (multi-modal) để cải thiện hiểu biết ngữ cảnh.",
        "citations": [{"label": "Nguồn 1", "value": "12", "snippet": "Kiến trúc đa phương thức giúp mô hình hiểu ngữ cảnh tốt hơn."}],
    },
]
_mock_idx = 0


# ── Models ────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str     # "user" hoặc "model"
    content: str

class AskRequest(BaseModel):
    message: str
    doc_id: Optional[str] = None
    notebook_id: Optional[str] = None
    history: List[HistoryMessage] = []


# ── System prompt chuẩn ───────────────────────────────────────────────────────

def build_system_prompt(context: str) -> str:
    return f"""Bạn là trợ lý học tập thông minh của trường Đại học Thủ Dầu Một (TDMU).

NHIỆM VỤ: Trả lời câu hỏi của sinh viên dựa trên tài liệu học tập được cung cấp.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc
- Trích dẫn nguồn bằng số [1], [2], [3]... sau mỗi luận điểm quan trọng
- Nếu tài liệu không đề cập → nói rõ: "Tài liệu không đề cập đến vấn đề này"
- Không bịa thêm thông tin ngoài tài liệu
- Nhớ và liên kết với các câu hỏi trước trong cuộc hội thoại
- Trả lời ngắn gọn, súc tích — ưu tiên bullet point khi liệt kê

TÀI LIỆU THAM KHẢO:
{context}"""


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(req: AskRequest, user: dict = Depends(get_current_user)):
    question = req.message.strip()
    if not question:
        raise HTTPException(400, "Câu hỏi không được để trống")

    uid = user["uid"]
    cache_key = req.notebook_id or req.doc_id or "all"

    # 1. Kiểm tra cache (chỉ cache khi không có history để tránh sai context)
    if not req.history:
        cached = get_cached(question, cache_key)
        if cached:
            return cached

    # 2. Embed câu hỏi
    try:
        q_embedding = await embed_query(question)
    except Exception as e:
        print(f"[ask] Embed error: {e}")
        return _mock_response()

    # 3. Tìm chunks qua pgvector (ưu tiên: notebook > doc > tất cả)
    sb = get_supabase()
    try:
        rpc_params: dict = {
            "query_embedding": q_embedding,
            "match_count": TOP_K,
            "user_id_filter": uid,
        }
        if req.notebook_id:
            rpc_name = "match_chunks_by_notebook"
            rpc_params["notebook_id_filter"] = req.notebook_id
        elif req.doc_id:
            rpc_name = "match_chunks_by_doc"
            rpc_params["doc_id_filter"] = req.doc_id
        else:
            rpc_name = "match_chunks"

        res = await asyncio.to_thread(
            lambda: sb.rpc(rpc_name, rpc_params).execute()
        )
        chunks = res.data or []
    except Exception as e:
        print(f"[ask] Supabase error: {e}")
        return _mock_response()

    if not chunks:
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan trong tài liệu. Bạn thử hỏi theo cách khác hoặc kiểm tra lại tài liệu đã upload.",
            "citations": []
        }

    # 4. Xây context + citations với snippet
    context_parts = []
    for i, c in enumerate(chunks):
        content = c.get("content", "")
        context_parts.append(f"[{i+1}] {content}")
    context = "\n---\n".join(context_parts)[:MAX_CONTEXT_CHARS]

    citations = [
        {
            "label": f"Nguồn {i+1}",
            "value": str(c.get("page_num", "?")),
            "snippet": c.get("content", "")[:200],    # 👈 thêm snippet
            "filename": c.get("filename", ""),          # 👈 thêm filename
        }
        for i, c in enumerate(chunks)
    ]

    # 5. Build Gemini chat với conversation history
    system_prompt = build_system_prompt(context)

    # Chuyển history Flutter → format Gemini Contents
    gemini_history = []
    for msg in req.history[-8:]:   # lấy 8 tin gần nhất
        gemini_history.append(
            genai.types.Content(
                role=msg.role,          # "user" hoặc "model"
                parts=[genai.types.Part(text=msg.content)]
            )
        )

    # 6. Gọi Gemini với chat history
    try:
        # Tạo chat session với history
        chat_session = _client.chats.create(
            model=GENERATE_MODEL,
            history=gemini_history,
        )

        # Gửi system prompt + câu hỏi hiện tại
        full_question = f"{system_prompt}\n\nCÂU HỎI: {question}"

        response = await asyncio.to_thread(
            lambda: chat_session.send_message(full_question)
        )
        answer = response.text.strip()

    except Exception as e:
        print(f"[ask] Gemini error: {e}")
        # Fallback về cách cũ nếu chat API lỗi
        try:
            prompt = f"{system_prompt}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
            response = await asyncio.to_thread(
                lambda: _client.models.generate_content(
                    model=GENERATE_MODEL,
                    contents=prompt,
                )
            )
            answer = response.text.strip()
        except Exception as e2:
            print(f"[ask] Gemini fallback error: {e2}")
            return _mock_response()

    result = {"answer": answer, "citations": citations}

    # Cache chỉ khi không có history (câu hỏi độc lập)
    if not req.history:
        set_cached(question, cache_key, result)

    return result


def _mock_response() -> dict:
    global _mock_idx
    mock = MOCK_QA[_mock_idx % len(MOCK_QA)]
    _mock_idx += 1
    return mock