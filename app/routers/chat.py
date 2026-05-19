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

TOP_K = 5
MAX_CONTEXT_CHARS = 4000

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
GENERATE_MODEL = "gemini-2.5-flash"

_MOCK_RESPONSE = {
    "answer": "Hệ thống đang bảo trì. Vui lòng thử lại sau.",
    "citations": [],
}


# ── Models ────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str    # "user" hoặc "model"
    content: str


class AskRequest(BaseModel):
    message: str
    doc_id: Optional[str] = None
    notebook_id: Optional[str] = None
    history: List[HistoryMessage] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_system_prompt(context: str, has_specific_source: bool) -> str:
    scope = (
        "dựa trên tài liệu/notebook mà người dùng đã chọn"
        if has_specific_source
        else "dựa trên toàn bộ tài liệu của người dùng trong hệ thống"
    )
    return f"""Bạn là trợ lý học tập thông minh của trường Đại học Thủ Dầu Một (TDMU).

NHIỆM VỤ: Trả lời câu hỏi của sinh viên {scope}.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc
- Trích dẫn nguồn bằng số [1], [2], [3]... sau mỗi luận điểm quan trọng
- Nếu tài liệu không đề cập → nói rõ: "Tài liệu không đề cập đến vấn đề này"
- Không bịa thêm thông tin ngoài tài liệu
- Nhớ và liên kết với các câu hỏi trước trong cuộc hội thoại
- Trả lời ngắn gọn, súc tích — ưu tiên bullet point khi liệt kê

TÀI LIỆU THAM KHẢO:
{context}"""


def _build_gemini_history(history: List[HistoryMessage]) -> list:
    """Chuyển history Flutter → Gemini Contents.

    Gemini yêu cầu:
    - Chỉ nhận role "user" hoặc "model"
    - History phải bắt đầu bằng role "user"
    - Các role phải xen kẽ user/model
    """
    valid_roles = {"user", "model"}
    items = []
    for msg in history[-8:]:
        role = msg.role if msg.role in valid_roles else "user"
        items.append(genai.types.Content(
            role=role,
            parts=[genai.types.Part(text=msg.content)],
        ))

    # Đảm bảo bắt đầu bằng "user"
    while items and items[0].role != "user":
        items.pop(0)

    return items


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(req: AskRequest, user: dict = Depends(get_current_user)):
    question = req.message.strip()
    if not question:
        raise HTTPException(400, "Câu hỏi không được để trống")

    uid = user["uid"]
    source_key = req.notebook_id or req.doc_id or "all"

    # 1. Kiểm tra cache (chỉ cache câu hỏi đơn, không có history)
    if not req.history:
        cached = get_cached(uid, question, source_key)
        if cached:
            return cached

    # 2. Embed câu hỏi
    try:
        q_embedding = await embed_query(question)
    except Exception as e:
        print(f"[ask] Embed error: {e}")
        return _MOCK_RESPONSE

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
        print(f"[ask] Supabase RPC error ({rpc_name}): {e}")
        return _MOCK_RESPONSE

    if not chunks:
        return {
            "answer": (
                "Tôi không tìm thấy thông tin liên quan trong tài liệu.\n\n"
                "Có thể do:\n"
                "• Tài liệu chưa được xử lý xong (status: processing)\n"
                "• Nội dung câu hỏi chưa khớp với tài liệu — thử diễn đạt khác\n"
                "• Tài liệu chưa được upload — vào tab Tải lên để thêm tài liệu"
            ),
            "citations": [],
        }

    # 4. Xây context + citations
    context_parts = [f"[{i+1}] {c.get('content', '')}" for i, c in enumerate(chunks)]
    context = "\n---\n".join(context_parts)[:MAX_CONTEXT_CHARS]

    citations = [
        {
            "label": f"Nguồn {i+1}",
            "value": str(c.get("page_num", "?")),
            "snippet": c.get("content", "")[:200],
            "filename": c.get("filename", ""),
        }
        for i, c in enumerate(chunks)
    ]

    # 5. Xây system prompt + Gemini history
    has_specific = bool(req.notebook_id or req.doc_id)
    system_prompt = _build_system_prompt(context, has_specific)
    gemini_history = _build_gemini_history(req.history)

    # 6. Gọi Gemini — dùng system_instruction thay vì nhúng vào user message
    answer = await _call_gemini(system_prompt, question, gemini_history)
    if answer is None:
        return _MOCK_RESPONSE

    result = {"answer": answer, "citations": citations}

    if not req.history:
        set_cached(uid, question, source_key, result)

    return result


async def _call_gemini(
    system_prompt: str,
    question: str,
    history: list,
) -> str | None:
    """Gọi Gemini với system instruction. Fallback về generate_content nếu lỗi."""
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
    )

    # Thử chat session (giữ lịch sử hội thoại)
    try:
        chat_session = _client.chats.create(
            model=GENERATE_MODEL,
            history=history,
            config=config,
        )
        response = await asyncio.to_thread(
            lambda: chat_session.send_message(question)
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ask] Gemini chat error: {e}")

    # Fallback: generate_content không cần history
    try:
        response = await asyncio.to_thread(
            lambda: _client.models.generate_content(
                model=GENERATE_MODEL,
                contents=question,
                config=config,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ask] Gemini fallback error: {e}")
        return None
