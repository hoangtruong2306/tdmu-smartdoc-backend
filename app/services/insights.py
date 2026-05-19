import asyncio
import json
import os
import re
from datetime import datetime, timezone

from google import genai

from app.services.database import get_supabase

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
GENERATE_MODEL = "gemini-2.5-flash"
MAX_TEXT_CHARS = 6000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def generate_notebook_insights(notebook_id: str, full_text: str) -> None:
    """Sinh tóm tắt và gợi ý câu hỏi cho notebook, lưu vào Supabase.

    Bug 4 fix: chỉ cập nhật nếu notebook chưa có summary (tránh ghi đè
    khi có nhiều tài liệu upload vào cùng notebook).
    """
    try:
        sb = get_supabase()

        # Kiểm tra notebook đã có summary chưa — nếu rồi thì bỏ qua
        existing = await asyncio.to_thread(
            lambda: sb.table("notebooks")
            .select("id, summary")
            .eq("id", notebook_id)
            .execute()
        )
        if not existing.data:
            return  # notebook không tồn tại
        if existing.data[0].get("summary", "").strip():
            return  # đã có summary, không ghi đè

        text_sample = full_text[:MAX_TEXT_CHARS]

        prompt = f"""Phân tích đoạn văn bản tài liệu học tập dưới đây và trả về JSON theo đúng format.

VĂN BẢN:
{text_sample}

Trả về JSON với cấu trúc sau (không thêm markdown, chỉ JSON thuần):
{{
  "summary": "Tóm tắt ngắn gọn nội dung chính của tài liệu (2-4 câu, tiếng Việt)",
  "suggestions": [
    "Câu hỏi gợi ý 1 về tài liệu?",
    "Câu hỏi gợi ý 2 về tài liệu?",
    "Câu hỏi gợi ý 3 về tài liệu?",
    "Câu hỏi gợi ý 4 về tài liệu?",
    "Câu hỏi gợi ý 5 về tài liệu?"
  ]
}}"""

        response = await asyncio.to_thread(
            lambda: _client.models.generate_content(
                model=GENERATE_MODEL,
                contents=prompt,
            )
        )
        raw = response.text.strip()

        # Tách JSON ra nếu bị bọc trong ```json ... ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        json_str = json_match.group(1).strip() if json_match else raw

        data = json.loads(json_str)
        summary = str(data.get("summary", "")).strip()
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []

        # Bug 1 fix: dùng ISO timestamp thật
        await asyncio.to_thread(
            lambda: sb.table("notebooks")
            .update({
                "summary": summary,
                "suggestions": suggestions,
                "updated_at": _now_iso(),
            })
            .eq("id", notebook_id)
            .execute()
        )

    except Exception as e:
        print(f"[insights] Lỗi sinh insights cho notebook {notebook_id}: {e}")
