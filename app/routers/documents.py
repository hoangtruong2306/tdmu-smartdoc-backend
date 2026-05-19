import asyncio
import uuid
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.auth import get_current_user
from app.services.database import get_supabase
from app.services.extractor import extract_chunks
from app.services.embeddings import embed_texts
from app.services.insights import generate_notebook_insights

router = APIRouter()

ALLOWED_TYPES = {"pdf", "txt"}
MAX_FILE_MB = 20


def _is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False


async def _embed_and_store(
    doc_id: str,
    uid: str,
    chunks,
    ext: str,
    notebook_id: Optional[str] = None,
):
    """Background task: embed chunks và lưu vào Supabase."""
    try:
        sb = get_supabase()
        texts = [c.content for c in chunks]
        embeddings = await embed_texts(texts)

        rows = [
            {
                "document_id": doc_id,
                "user_id": uid,
                "content": chunks[i].content,
                "page_num": chunks[i].page_num,
                "embedding": embeddings[i],
                "notebook_id": notebook_id,
            }
            for i in range(len(chunks))
        ]
        for i in range(0, len(rows), 100):
            await asyncio.to_thread(
                lambda r=rows[i : i + 100]: sb.table("chunks").insert(r).execute()
            )

        await asyncio.to_thread(
            lambda: sb.table("documents")
            .update({"status": "ready"})
            .eq("id", doc_id)
            .execute()
        )

        if notebook_id:
            full_text = " ".join(c.content for c in chunks)
            await generate_notebook_insights(notebook_id, full_text)

    except Exception as e:
        print(f"[embed_and_store] Lỗi doc {doc_id}: {e}")


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    notebook_id: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Chỉ hỗ trợ: {', '.join(ALLOWED_TYPES)}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File vượt quá {MAX_FILE_MB} MB")

    uid = user["uid"]

    # Bug 6 fix: validate UUID format
    if notebook_id is not None:
        if not _is_valid_uuid(notebook_id):
            raise HTTPException(400, "notebook_id không hợp lệ")

        # Bug 2 fix (Security): kiểm tra notebook thuộc về user hiện tại
        sb = get_supabase()
        nb_check = await asyncio.to_thread(
            lambda: sb.table("notebooks")
            .select("id")
            .eq("id", notebook_id)
            .eq("user_id", uid)
            .execute()
        )
        if not nb_check.data:
            raise HTTPException(403, "Notebook không tồn tại hoặc không có quyền truy cập")

    doc_id = str(uuid.uuid4())
    sb = get_supabase()

    chunks, page_count = extract_chunks(file_bytes, file.filename or "file.pdf")
    if not chunks:
        raise HTTPException(422, "Không trích xuất được nội dung từ file")

    await asyncio.to_thread(
        lambda: sb.table("documents").insert({
            "id": doc_id,
            "user_id": uid,
            "title": file.filename or "Tài liệu",
            "page_count": page_count,
            "type": ext,
            "status": "processing",
            "notebook_id": notebook_id,
        }).execute()
    )

    background_tasks.add_task(_embed_and_store, doc_id, uid, chunks, ext, notebook_id)

    return {
        "document_id": doc_id,
        "title": file.filename,
        "page_count": page_count,
        "chunks_count": len(chunks),
        "notebook_id": notebook_id,
        "status": "processing",
    }


@router.get("")
async def list_documents(user: dict = Depends(get_current_user)):
    uid = user["uid"]
    sb = get_supabase()
    res = await asyncio.to_thread(
        lambda: sb.table("documents")
        .select("id, title, page_count, type, created_at, status, notebook_id")
        .eq("user_id", uid)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []
