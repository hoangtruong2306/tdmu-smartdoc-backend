# =============================================================================
# QUIZ AI ROUTER — Sinh câu hỏi trắc nghiệm từ nội dung notebook
# =============================================================================
#
# LUỒNG NGHIỆP VỤ:
#   1. Client gửi POST /quiz/generate với notebook_id + num_questions + difficulty
#   2. Backend kiểm tra notebook thuộc user (ownership check)
#   3. Lấy tối đa 15 chunks từ Supabase (table "chunks", sắp xếp theo chunk_index)
#   4. Nối chunks thành context string (giới hạn 5000 ký tự)
#   5. Gọi Gemini 2.5 Flash → sinh JSON quiz theo prompt cấu trúc
#   6. Parse & validate JSON output:
#      - Phải có trường "questions" là list
#      - Mỗi question có đúng 4 options
#   7. Nếu bất kỳ bước 3-6 lỗi → trả mock quiz (không crash app)
#
# FALLBACK STRATEGY:
#   - Chunks rỗng → is_mock: true (notebook chưa có tài liệu)
#   - Gemini lỗi / JSON lỗi → is_mock: true (fallback graceful)
#   - Luôn trả HTTP 200 kèm is_mock flag để Flutter biết cần thông báo
#
# DIFFICULTY MAP:
#   easy   → nhớ & hiểu (Bloom level 1-2)
#   medium → vận dụng & phân tích (Bloom level 3-4)
#   hard   → đánh giá & tổng hợp (Bloom level 5-6)
# =============================================================================

import asyncio
import json
import os
import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from google import genai

from app.auth import get_current_user
from app.services.database import get_supabase

router = APIRouter()

# ── Gemini client (dùng chung với chat.py, không tạo lại) ─────────────────────
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
GENERATE_MODEL = "gemini-2.5-flash"

# ── Mock quiz dự phòng — trả khi notebook rỗng hoặc Gemini lỗi ───────────────
# Nội dung cố định về kinh tế vi mô (môn phổ biến ở TDMU)
# Mục đích: tránh màn hình trắng, cho UX liền mạch
MOCK_QUIZ = {
    "questions": [
        {
            "question": "Kinh tế học vi mô nghiên cứu điều gì?",
            "options": [
                "A. Toàn bộ nền kinh tế quốc gia",
                "B. Hành vi của các tác nhân kinh tế riêng lẻ",
                "C. Chính sách tiền tệ của ngân hàng trung ương",
                "D. Thương mại quốc tế giữa các quốc gia",
            ],
            "correct": "B",
            "explanation": "Kinh tế vi mô tập trung vào hành vi cá nhân: hộ gia đình, doanh nghiệp và thị trường cụ thể.",
        },
        {
            "question": "Quy luật cầu phát biểu rằng?",
            "options": [
                "A. Giá tăng thì cầu tăng",
                "B. Giá giảm thì cầu giảm",
                "C. Giá tăng thì lượng cầu giảm",
                "D. Giá và cầu không liên quan",
            ],
            "correct": "C",
            "explanation": "Quy luật cầu: khi giá tăng, người tiêu dùng mua ít hơn (ceteris paribus).",
        },
        {
            "question": "Thị trường cạnh tranh hoàn hảo có đặc điểm nào?",
            "options": [
                "A. Một người bán duy nhất",
                "B. Sản phẩm khác biệt",
                "C. Rào cản gia nhập cao",
                "D. Nhiều người mua và bán, sản phẩm đồng nhất",
            ],
            "correct": "D",
            "explanation": "Cạnh tranh hoàn hảo: nhiều người tham gia, sản phẩm đồng nhất, thông tin đầy đủ, tự do gia nhập.",
        },
        {
            "question": "Độ co giãn của cầu theo giá đo lường điều gì?",
            "options": [
                "A. Mức độ thay đổi của giá",
                "B. Mức độ phản ứng của lượng cầu khi giá thay đổi",
                "C. Tổng doanh thu của doanh nghiệp",
                "D. Chi phí sản xuất trung bình",
            ],
            "correct": "B",
            "explanation": "Co giãn cầu = %ΔQ / %ΔP, đo mức độ nhạy cảm của lượng cầu với sự thay đổi giá.",
        },
        {
            "question": "Chi phí cơ hội là gì?",
            "options": [
                "A. Chi phí mua nguyên vật liệu",
                "B. Chi phí thuê nhân công",
                "C. Giá trị của lựa chọn tốt nhất bị bỏ qua",
                "D. Tổng chi phí cố định và biến đổi",
            ],
            "correct": "C",
            "explanation": "Chi phí cơ hội là giá trị của phương án tốt nhất mà bạn từ bỏ khi đưa ra một quyết định.",
        },
    ]
}


# ── Models ────────────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    notebook_id: str
    num_questions: int = 5
    difficulty: Optional[str] = "medium"   # "easy" | "medium" | "hard"


# ── Helper: lấy chunks từ Supabase ───────────────────────────────────────────
#
# THUẬT TOÁN:
#   - Query bảng "chunks" theo notebook_id + user_id (ownership isolation)
#   - Sắp xếp theo chunk_index để giữ thứ tự logic tài liệu gốc
#   - Giới hạn `limit` chunks (mặc định 15) để tránh vượt token Gemini
#   - Nối các chunk bằng "\n\n", cắt bớt còn 5000 ký tự
#   - Trả "" nếu lỗi → caller sẽ fallback mock
#
async def _get_notebook_context(
    notebook_id: str,
    user_id: str,
    sb,
    limit: int = 15,
) -> str:
    """Lấy chunks từ Supabase theo notebook_id, sắp xếp theo page_num.

    Bảng chunks có các cột: document_id, user_id, content, page_num,
    embedding, notebook_id — KHÔNG có chunk_index.
    Dùng page_num để sắp xếp nội dung theo thứ tự trang tài liệu.
    """
    try:
        res = await asyncio.to_thread(
            lambda: sb.table("chunks")
                .select("content, page_num")      # chỉ lấy 2 trường cần thiết
                .eq("notebook_id", notebook_id)
                .eq("user_id", user_id)
                .order("page_num")                # sắp xếp theo trang (tồn tại trong schema)
                .limit(limit)                     # giới hạn token Gemini
                .execute()
        )
        chunks = res.data or []
        if not chunks:
            return ""
        # Nối chunks với dấu phân cách, cắt tối đa 5000 ký tự tránh vượt context
        return "\n\n".join(c["content"] for c in chunks)[:5000]
    except Exception as e:
        print(f"[quiz] Lỗi lấy chunks: {e}")
        return ""


# ── Helper: gọi Gemini sinh quiz JSON ────────────────────────────────────────
#
# THUẬT TOÁN:
#   1. Map difficulty string → mô tả mức độ nhận thức (Bloom's Taxonomy)
#   2. Xây prompt yêu cầu Gemini trả raw JSON (không markdown, không ```json```)
#   3. Gọi generate_content đồng bộ (wrap bằng asyncio.to_thread)
#   4. Dùng regex xóa markdown code fence nếu Gemini vẫn thêm vào
#   5. Parse JSON → trả dict Python
#   6. Throws: JSONDecodeError, ValueError → caller bắt → fallback mock
#
async def _generate_with_gemini(
    context: str,
    num_questions: int,
    difficulty: str,
) -> dict:
    # Ánh xạ difficulty → mô tả tiếng Việt theo thang Bloom
    difficulty_map = {
        "easy":   "cơ bản, nhớ và hiểu (Bloom level 1-2)",
        "medium": "vận dụng và phân tích (Bloom level 3-4)",
        "hard":   "đánh giá và tổng hợp, tình huống phức tạp (Bloom level 5-6)",
    }
    level_desc = difficulty_map.get(difficulty, "vận dụng và phân tích")

    # Prompt yêu cầu Gemini trả đúng JSON structure
    # Quan trọng: "không có markdown" để tránh phải strip ```json
    prompt = f"""Bạn là giảng viên đại học. Tạo {num_questions} câu hỏi trắc nghiệm mức độ {level_desc} từ tài liệu dưới đây.

YÊU CẦU:
- Câu hỏi rõ ràng, chính xác, bằng tiếng Việt
- Mỗi câu có đúng 4 lựa chọn A/B/C/D
- Chỉ 1 đáp án đúng
- Giải thích ngắn gọn tại sao đáp án đúng (1-2 câu)
- Không hỏi về thông tin nằm ngoài tài liệu

Trả về JSON hợp lệ (không có markdown, không có ký tự thừa):
{{
  "questions": [
    {{
      "question": "Nội dung câu hỏi?",
      "options": ["A. Lựa chọn A", "B. Lựa chọn B", "C. Lựa chọn C", "D. Lựa chọn D"],
      "correct": "A",
      "explanation": "Giải thích ngắn gọn"
    }}
  ]
}}

TÀI LIỆU:
{context}"""

    # Gọi Gemini đồng bộ trong thread pool (không block event loop)
    response = await asyncio.to_thread(
        lambda: _client.models.generate_content(
            model=GENERATE_MODEL,
            contents=prompt,
        )
    )

    raw = response.text.strip()

    # Strip markdown code fence phòng trường hợp Gemini không tuân thủ
    # Pattern: ```json ... ``` hoặc ``` ... ```
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()

    return json.loads(raw)  # Throws JSONDecodeError nếu JSON không hợp lệ


# ── Endpoint chính: POST /quiz/generate ──────────────────────────────────────
#
# LUỒNG XỬ LÝ:
#   1. Validate num_questions (1-20)
#   2. Kiểm tra notebook ownership (404 nếu không tìm thấy)
#   3. Lấy context từ chunks (rỗng → mock)
#   4. Gọi Gemini → parse JSON → validate cấu trúc
#   5. Bất kỳ exception nào ở bước 4 → fallback mock (không raise 500)
#
# OWNERSHIP CHECK:
#   Lọc theo cả notebook.id VÀ notebook.user_id → tránh user A truy cập notebook B
#
@router.post("/generate")
async def generate_quiz(
    req: QuizRequest,
    user: dict = Depends(get_current_user),
):
    # Validate đầu vào
    if req.num_questions < 1 or req.num_questions > 20:
        raise HTTPException(400, "Số câu hỏi phải từ 1 đến 20")

    sb = get_supabase()
    uid = user["uid"]

    # ── Bước 1: Kiểm tra notebook ownership ──────────────────────────────────
    try:
        nb_check = await asyncio.to_thread(
            lambda: sb.table("notebooks")
                .select("id, name")
                .eq("id", req.notebook_id)
                .eq("user_id", uid)      # ownership isolation
                .execute()
        )
        if not nb_check.data:
            raise HTTPException(404, "Notebook không tồn tại hoặc không thuộc về bạn")
        notebook_name = nb_check.data[0].get("name", "Notebook")
    except HTTPException:
        raise  # re-raise 404 — không bắt thành 500
    except Exception as e:
        print(f"[quiz] Lỗi kiểm tra notebook: {e}")
        raise HTTPException(500, "Lỗi kết nối database")

    # ── Bước 2: Lấy context từ chunks ────────────────────────────────────────
    context = await _get_notebook_context(req.notebook_id, uid, sb)

    # Notebook chưa có tài liệu → trả mock quiz với cờ is_mock: true
    if not context:
        print(f"[quiz] Notebook {req.notebook_id[:8]}... rỗng → trả mock quiz")
        mock = dict(MOCK_QUIZ)
        mock["questions"] = mock["questions"][: req.num_questions]
        mock["notebook_name"] = notebook_name
        mock["is_mock"] = True
        return mock

    # ── Bước 3: Gọi Gemini + fallback khi lỗi ────────────────────────────────
    try:
        result = await _generate_with_gemini(
            context,
            req.num_questions,
            req.difficulty or "medium",
        )

        # Validate cấu trúc JSON trả về
        questions = result.get("questions", [])
        if not questions:
            raise ValueError("Gemini trả về list câu hỏi rỗng")
        for q in questions:
            if len(q.get("options", [])) != 4:
                raise ValueError(f"Câu hỏi thiếu options: {q.get('question', '')[:50]}")
            if q.get("correct", "") not in ("A", "B", "C", "D"):
                raise ValueError(f"correct không hợp lệ: {q.get('correct')}")

        result["notebook_name"] = notebook_name
        result["is_mock"] = False
        return result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # JSON parse lỗi hoặc cấu trúc sai → graceful fallback
        print(f"[quiz] Parse/validate error: {e} → fallback mock")

    except Exception as e:
        # Lỗi Gemini API (quota, network...) → graceful fallback
        print(f"[quiz] Gemini error: {e} → fallback mock")

    # Fallback: trả mock nhưng vẫn gắn notebook_name thật
    mock = dict(MOCK_QUIZ)
    mock["questions"] = mock["questions"][: req.num_questions]
    mock["notebook_name"] = notebook_name
    mock["is_mock"] = True
    return mock
