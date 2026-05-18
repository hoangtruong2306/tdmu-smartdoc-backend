import asyncio
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.auth import get_current_user
from app.services.database import get_supabase
from app.services.extractor import extract_chunks
from app.services.embeddings import embed_texts

router = APIRouter()

ALLOWED_TYPES = {"pdf", "txt"}
MAX_FILE_MB = 20


async def _embed_and_store(doc_id: str, uid: str, chunks, ext: str):
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
            }
            for i in range(len(chunks))
        ]
        for i in range(0, len(rows), 100):
            await asyncio.to_thread(
                lambda r=rows[i : i + 100]: sb.table("chunks").insert(r).execute()
            )

        # Cập nhật trạng thái tài liệu → ready
        await asyncio.to_thread(
            lambda: sb.table("documents")
            .update({"status": "ready"})
            .eq("id", doc_id)
            .execute()
        )
    except Exception as e:
        print(f"[embed_and_store] Lỗi doc {doc_id}: {e}")


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Chỉ hỗ trợ: {', '.join(ALLOWED_TYPES)}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File vượt quá {MAX_FILE_MB} MB")

    uid = user["uid"]
    doc_id = str(uuid.uuid4())
    sb = get_supabase()

    # 1. Trích xuất + chia chunks (nhanh, sync)
    chunks, page_count = extract_chunks(file_bytes, file.filename or "file.pdf")
    if not chunks:
        raise HTTPException(422, "Không trích xuất được nội dung từ file")

    # 2. Lưu metadata ngay (status=processing)
    await asyncio.to_thread(
        lambda: sb.table("documents").insert({
            "id": doc_id,
            "user_id": uid,
            "title": file.filename or "Tài liệu",
            "page_count": page_count,
            "type": ext,
            "status": "processing",
        }).execute()
    )

    # 3. Embed + lưu chunks → background (không block response)
    background_tasks.add_task(_embed_and_store, doc_id, uid, chunks, ext)

    # 4. Trả về ngay cho client (~2-3s thay vì 30-60s)
    return {
        "document_id": doc_id,
        "title": file.filename,
        "page_count": page_count,
        "chunks_count": len(chunks),
        "status": "processing",
    }


@router.get("")
async def list_documents(user: dict = Depends(get_current_user)):
    uid = user["uid"]
    sb = get_supabase()
    res = await asyncio.to_thread(
        lambda: sb.table("documents")
        .select("id, title, page_count, type, created_at, status")
        .eq("user_id", uid)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []
