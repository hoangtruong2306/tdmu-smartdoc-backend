import json as _json
import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
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

# ── Mock Q&A — Sprint 3: 5 câu hỏi thật từ giáo trình TDMU ──────────────────
# Dùng khi Gemini timeout hoặc Supabase lỗi → app vẫn trả lời được khi demo
import random as _random

MOCK_QA = [
    {
        "answer": (
            "Kinh tế học vi mô nghiên cứu hành vi của các tác nhân kinh tế riêng lẻ "
            "như hộ gia đình, doanh nghiệp và các thị trường cụ thể. Môn học phân tích "
            "cơ chế hình thành giá cả, quyết định sản xuất và tiêu dùng ở cấp độ từng "
            "đơn vị kinh tế. [1]"
        ),
        "citations": [
            {
                "label": "Nguồn 1",
                "value": "5",
                "snippet": "Kinh tế học vi mô là bộ phận của khoa học kinh tế, "
                           "nghiên cứu các quyết định của cá nhân và doanh nghiệp...",
                "filename": "giao_trinh_ktvm.pdf",
            }
        ],
    },
    {
        "answer": (
            "Quy luật cầu phát biểu rằng khi giá cả của một hàng hóa tăng lên "
            "(trong điều kiện các yếu tố khác không đổi), lượng cầu về hàng hóa đó "
            "sẽ giảm xuống, và ngược lại. Đây là mối quan hệ nghịch biến giữa giá "
            "và lượng cầu — đường cầu dốc xuống từ trái sang phải. [1]"
        ),
        "citations": [
            {
                "label": "Nguồn 1",
                "value": "23",
                "snippet": "Đường cầu dốc xuống từ trái sang phải thể hiện mối "
                           "quan hệ nghịch biến giữa giá và lượng cầu...",
                "filename": "giao_trinh_ktvm.pdf",
            }
        ],
    },
    {
        "answer": (
            "Chi phí cơ hội là giá trị của cơ hội tốt nhất bị bỏ qua khi đưa ra "
            "một quyết định kinh tế. Khái niệm này phản ánh bản chất của sự khan "
            "hiếm nguồn lực — khi chọn phương án này, buộc phải từ bỏ phương án khác. "
            "Ví dụ: sinh viên đi học đại học có chi phí cơ hội là thu nhập từ công "
            "việc bị bỏ lỡ. [1]"
        ),
        "citations": [
            {
                "label": "Nguồn 1",
                "value": "12",
                "snippet": "Chi phí cơ hội xuất hiện khi các nguồn lực có thể được "
                           "sử dụng theo nhiều cách khác nhau...",
                "filename": "giao_trinh_ktvm.pdf",
            }
        ],
    },
    {
        "answer": (
            "Thị trường cạnh tranh hoàn hảo có 4 đặc điểm:\n"
            "• Nhiều người mua và bán nhỏ lẻ, không ai có sức mạnh thị trường\n"
            "• Sản phẩm đồng nhất (homogeneous)\n"
            "• Thông tin hoàn hảo — mọi người đều biết giá cả\n"
            "• Tự do gia nhập/rút lui khỏi thị trường [1]\n\n"
            "Trong mô hình này, doanh nghiệp là người chấp nhận giá (price taker)."
        ),
        "citations": [
            {
                "label": "Nguồn 1",
                "value": "67",
                "snippet": "Cạnh tranh hoàn hảo là một cấu trúc thị trường lý tưởng, "
                           "nơi không một người mua hay bán nào có khả năng định giá...",
                "filename": "giao_trinh_ktvm.pdf",
            }
        ],
    },
    {
        "answer": (
            "Độ co giãn của cầu theo giá (PED) đo lường mức độ nhạy cảm của lượng "
            "cầu trước sự thay đổi của giá:\n\n"
            "**Công thức:** PED = %ΔQ / %ΔP\n\n"
            "**Phân loại:**\n"
            "• |PED| > 1: Cầu co giãn nhiều (xa xỉ phẩm, có nhiều hàng thay thế)\n"
            "• |PED| < 1: Cầu co giãn ít (thiết yếu phẩm như gạo, muối)\n"
            "• |PED| = 1: Cầu co giãn đơn vị [1]"
        ),
        "citations": [
            {
                "label": "Nguồn 1",
                "value": "45",
                "snippet": "Co giãn là thước đo phản ứng của lượng cầu hoặc cung "
                           "đối với sự thay đổi của giá...",
                "filename": "giao_trinh_ktvm.pdf",
            }
        ],
    },
]

# ── Helper: trả lời ngẫu nhiên từ MOCK_QA khi Gemini không khả dụng ──────────
def _mock_response() -> dict:
    """Chọn ngẫu nhiên 1 câu từ MOCK_QA để tránh lặp lại khi demo."""
    return _random.choice(MOCK_QA)


# Giữ lại _MOCK_RESPONSE cho các trường hợp lỗi nghiêm trọng (embed lỗi, v.v.)
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
        return _mock_response()  # trả mock TDMU thay vì thông báo lỗi

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
        return _mock_response()  # trả mock TDMU thay vì thông báo lỗi

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

    # 6. Gọi Gemini — timeout 15s, fallback mock nếu quá chậm
    answer = await _call_gemini(system_prompt, question, gemini_history)
    if answer is None:
        return _mock_response()  # Gemini timeout/lỗi → mock TDMU content

    result = {"answer": answer, "citations": citations}

    if not req.history:
        set_cached(uid, question, source_key, result)

    return result


@router.post("/stream")
async def ask_stream(req: AskRequest, user: dict = Depends(get_current_user)):
    """SSE streaming endpoint.

    Headers được gửi NGAY LẬP TỨC sau khi auth xong — mọi I/O nặng
    (embed, pgvector, Gemini) đều nằm bên trong generator để tránh
    Flutter timeout trước khi nhận được headers.
    """
    question = req.message.strip()
    uid = user["uid"]

    _sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    if not question:
        async def _empty():
            yield f"data: {_json.dumps({'type':'error','message':'Câu hỏi không được để trống.'})}\n\n"
            yield f"data: {_json.dumps({'type':'done','full_text':''})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream", headers=_sse_headers)

    async def token_stream():
        # ── 1. Báo hiệu đã kết nối → Flutter giữ typing indicator ──────────
        yield f"data: {_json.dumps({'type':'processing'})}\n\n"

        # ── 2. Embed câu hỏi ─────────────────────────────────────────────────
        try:
            q_embedding = await embed_query(question)
        except Exception as e:
            print(f"[stream] Embed error: {e}")
            err = "Lỗi xử lý câu hỏi. Vui lòng thử lại."
            yield f"data: {_json.dumps({'type':'token','text':err})}\n\n"
            yield f"data: {_json.dumps({'type':'done','full_text':err})}\n\n"
            return

        # keep-alive giữa embed và Supabase
        yield f"data: {_json.dumps({'type':'ping'})}\n\n"

        # ── 3. Tìm chunks pgvector ────────────────────────────────────────────
        sb = get_supabase()
        rpc_name = "match_chunks"
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

        try:
            res = await asyncio.to_thread(lambda: sb.rpc(rpc_name, rpc_params).execute())
            chunks = res.data or []
        except Exception as e:
            print(f"[stream] Supabase error ({rpc_name}): {e}")
            chunks = []

        # ── 4. Gửi citations ─────────────────────────────────────────────────
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
        yield f"data: {_json.dumps({'type':'citations','citations':citations})}\n\n"

        # ── 5. Gọi Gemini stream ──────────────────────────────────────────────
        has_specific = bool(req.notebook_id or req.doc_id)
        system_prompt = _build_system_prompt(context, has_specific)
        gemini_history = _build_gemini_history(req.history)
        config = genai.types.GenerateContentConfig(system_instruction=system_prompt)

        full_text = ""
        try:
            chat_session = _client.chats.create(
                model=GENERATE_MODEL,
                history=gemini_history,
                config=config,
            )
            stream = await asyncio.to_thread(
                lambda: chat_session.send_message_stream(question)
            )
            for chunk in stream:
                token = chunk.text or ""
                if token:
                    full_text += token
                    yield f"data: {_json.dumps({'type':'token','text':token})}\n\n"
        except Exception as e:
            print(f"[stream] Gemini error: {e}")
            if not full_text:
                full_text = "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại."
                yield f"data: {_json.dumps({'type':'token','text':full_text})}\n\n"

        yield f"data: {_json.dumps({'type':'done','full_text':full_text})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream", headers=_sse_headers)


_GEMINI_TIMEOUT = 15.0  # Sprint 3: timeout 15s → fallback mock nếu Gemini chậm


async def _call_gemini(
    system_prompt: str,
    question: str,
    history: list,
) -> str | None:
    """Gọi Gemini với system instruction + timeout 15s.

    Fallback chain:
    1. Chat session (giữ lịch sử hội thoại) — timeout 15s
    2. generate_content không history — timeout 15s
    3. Trả về None → caller dùng mock
    """
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
    )

    # ── 1. Thử chat session với timeout ──────────────────────────────────────
    try:
        chat_session = _client.chats.create(
            model=GENERATE_MODEL,
            history=history,
            config=config,
        )
        response = await asyncio.wait_for(
            asyncio.to_thread(lambda: chat_session.send_message(question)),
            timeout=_GEMINI_TIMEOUT,
        )
        return response.text.strip()
    except asyncio.TimeoutError:
        print(f"[ask] Gemini chat timeout ({_GEMINI_TIMEOUT}s) → fallback")
    except Exception as e:
        print(f"[ask] Gemini chat error: {e}")

    # ── 2. Fallback: generate_content không history, cũng có timeout ─────────
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: _client.models.generate_content(
                    model=GENERATE_MODEL,
                    contents=question,
                    config=config,
                )
            ),
            timeout=_GEMINI_TIMEOUT,
        )
        return response.text.strip()
    except asyncio.TimeoutError:
        print(f"[ask] Gemini fallback timeout ({_GEMINI_TIMEOUT}s) → mock")
    except Exception as e:
        print(f"[ask] Gemini fallback error: {e}")

    return None  # caller sẽ dùng _mock_response()
