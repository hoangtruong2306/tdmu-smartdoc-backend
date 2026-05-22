import asyncio
import uuid
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

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
    notebook_id: str,  # REQUIRED — document phải thuộc notebook
):
    """Background task: embed chunks và lưu vào Supabase.

    Document embedding và chunks sẽ được gắn với notebook_id.
    Cập nhật status='failed' nếu lỗi để UI biết và user có thể upload lại.
    """
    sb = get_supabase()
    try:
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
        try:
            await asyncio.to_thread(
                lambda: sb.table("documents")
                .update({"status": "failed"})
                .eq("id", doc_id)
                .execute()
            )
        except Exception as e2:
            print(f"[embed_and_store] Không thể cập nhật status failed cho {doc_id}: {e2}")


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    notebook_id: str = Form(...),  # REQUIRED — mỗi file phải thuộc 1 notebook
    user: dict = Depends(get_current_user),
):
    """Upload PDF/Slide vào notebook cụ thể.

    notebook_id REQUIRED — đảm bảo document isolation.
    """
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Chỉ hỗ trợ: {', '.join(ALLOWED_TYPES)}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File vượt quá {MAX_FILE_MB} MB")

    uid = user["uid"]

    # Validate notebook_id format
    if not _is_valid_uuid(notebook_id):
        raise HTTPException(400, "notebook_id không hợp lệ (phải là UUID)")

    # Kiểm tra notebook tồn tại và user có quyền truy cập
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

    chunks, page_count = extract_chunks(file_bytes, file.filename or "file.pdf")
    if not chunks:
        raise HTTPException(422, "Không trích xuất được nội dung từ file")

    # Insert document với notebook_id (REQUIRED)
    await asyncio.to_thread(
        lambda: sb.table("documents").insert({
            "id": doc_id,
            "user_id": uid,
            "title": file.filename or "Tài liệu",
            "page_count": page_count,
            "type": ext,
            "status": "processing",
            "notebook_id": notebook_id,  # luôn có giá trị
        }).execute()
    )

    # Background task: embed chunks và lưu vào database với notebook context
    background_tasks.add_task(_embed_and_store, doc_id, uid, chunks, notebook_id)

    return {
        "document_id": doc_id,
        "title": file.filename,
        "page_count": page_count,
        "chunks_count": len(chunks),
        "notebook_id": notebook_id,
        "status": "processing",
    }


@router.get("")
async def list_documents(
    notebook_id: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """Lấy danh sách documents của user.

    Query params:
    - notebook_id (optional): lọc theo notebook cụ thể
    """
    uid = user["uid"]
    sb = get_supabase()

    # Xây dựng query
    query = sb.table("documents") \
        .select("id, title, page_count, type, created_at, status, notebook_id") \
        .eq("user_id", uid)

    # Filter by notebook nếu được cung cấp
    if notebook_id is not None:
        if not _is_valid_uuid(notebook_id):
            raise HTTPException(400, "notebook_id không hợp lệ (phải là UUID)")
        query = query.eq("notebook_id", notebook_id)

    # Execute
    res = await asyncio.to_thread(
        lambda: query.order("created_at", desc=True).execute()
    )
    return res.data or []


# ── Assign existing documents to a notebook ──────────────────────────────────

class AssignDocumentsRequest(BaseModel):
    doc_ids: List[str]
    notebook_id: str


@router.post("/assign")
async def assign_documents_to_notebook(
    body: AssignDocumentsRequest,
    user: dict = Depends(get_current_user),
):
    """Gán danh sách tài liệu đã upload vào một notebook cụ thể.

    Cập nhật notebook_id cho cả documents lẫn chunks để RAG search vẫn hoạt động.
    Chỉ cho phép gán documents thuộc chính user đang đăng nhập.
    """
    uid = user["uid"]

    if not body.doc_ids:
        raise HTTPException(400, "doc_ids không được rỗng")
    if not _is_valid_uuid(body.notebook_id):
        raise HTTPException(400, "notebook_id không hợp lệ (phải là UUID)")

    # Validate tất cả doc_ids là UUID hợp lệ
    for doc_id in body.doc_ids:
        if not _is_valid_uuid(doc_id):
            raise HTTPException(400, f"doc_id không hợp lệ: {doc_id}")

    sb = get_supabase()

    # Kiểm tra notebook tồn tại và thuộc user này
    nb_check = await asyncio.to_thread(
        lambda: sb.table("notebooks")
        .select("id")
        .eq("id", body.notebook_id)
        .eq("user_id", uid)
        .execute()
    )
    if not nb_check.data:
        raise HTTPException(403, "Notebook không tồn tại hoặc không có quyền truy cập")

    # Update documents: chỉ update docs thuộc user này (bảo mật)
    assigned: List[str] = []
    for doc_id in body.doc_ids:
        result = await asyncio.to_thread(
            lambda d=doc_id: sb.table("documents")
            .update({"notebook_id": body.notebook_id})
            .eq("id", d)
            .eq("user_id", uid)
            .execute()
        )
        if result.data:
            assigned.append(doc_id)

    # Cập nhật chunks để match_chunks_by_notebook RPC hoạt động đúng
    for doc_id in assigned:
        await asyncio.to_thread(
            lambda d=doc_id: sb.table("chunks")
            .update({"notebook_id": body.notebook_id})
            .eq("document_id", d)
            .eq("user_id", uid)
            .execute()
        )

    return {
        "assigned_count": len(assigned),
        "assigned": assigned,
        "notebook_id": body.notebook_id,
    }
