import asyncio
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.auth import get_current_user
from app.services.database import get_supabase

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False


class NotebookCreate(BaseModel):
    name: str
    color: Optional[str] = "#6750A4"


class NotebookUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


@router.get("")
async def list_notebooks(user: dict = Depends(get_current_user)):
    uid = user["uid"]
    sb = get_supabase()
    res = await asyncio.to_thread(
        lambda: sb.table("notebooks")
        .select("id, name, color, summary, suggestions, created_at, updated_at")
        .eq("user_id", uid)
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


@router.post("")
async def create_notebook(body: NotebookCreate, user: dict = Depends(get_current_user)):
    uid = user["uid"]
    if not body.name.strip():
        raise HTTPException(400, "Tên notebook không được để trống")

    nb_id = str(uuid.uuid4())
    sb = get_supabase()
    res = await asyncio.to_thread(
        lambda: sb.table("notebooks").insert({
            "id": nb_id,
            "user_id": uid,
            "name": body.name.strip(),
            "color": body.color or "#6750A4",
        }).execute()
    )
    data = res.data or []
    if not data:
        raise HTTPException(500, "Tạo notebook thất bại")
    return data[0]


@router.patch("/{notebook_id}")
async def update_notebook(
    notebook_id: str,
    body: NotebookUpdate,
    user: dict = Depends(get_current_user),
):
    # Bug 5 fix: validate ít nhất 1 field được cập nhật
    if body.name is None and body.color is None:
        raise HTTPException(400, "Cần cung cấp ít nhất name hoặc color để cập nhật")

    # Bug 6 fix: validate UUID format
    if not _is_valid_uuid(notebook_id):
        raise HTTPException(400, "notebook_id không hợp lệ")

    uid = user["uid"]
    sb = get_supabase()

    existing = await asyncio.to_thread(
        lambda: sb.table("notebooks")
        .select("id")
        .eq("id", notebook_id)
        .eq("user_id", uid)
        .execute()
    )
    if not existing.data:
        raise HTTPException(404, "Notebook không tồn tại")

    # Bug 1 fix: dùng ISO timestamp thật, không phải string "now()"
    updates: dict = {"updated_at": _now_iso()}
    if body.name is not None:
        updates["name"] = body.name.strip()
    if body.color is not None:
        updates["color"] = body.color

    res = await asyncio.to_thread(
        lambda: sb.table("notebooks")
        .update(updates)
        .eq("id", notebook_id)
        .eq("user_id", uid)
        .execute()
    )
    # Bug 3 fix: trả về object có nghĩa thay vì {}
    data = res.data or []
    return data[0] if data else {"id": notebook_id, **updates}


@router.delete("/{notebook_id}")
async def delete_notebook(
    notebook_id: str,
    user: dict = Depends(get_current_user),
):
    if not _is_valid_uuid(notebook_id):
        raise HTTPException(400, "notebook_id không hợp lệ")

    uid = user["uid"]
    sb = get_supabase()

    existing = await asyncio.to_thread(
        lambda: sb.table("notebooks")
        .select("id")
        .eq("id", notebook_id)
        .eq("user_id", uid)
        .execute()
    )
    if not existing.data:
        raise HTTPException(404, "Notebook không tồn tại")

    await asyncio.to_thread(
        lambda: sb.table("notebooks")
        .delete()
        .eq("id", notebook_id)
        .eq("user_id", uid)
        .execute()
    )
    return {"deleted": notebook_id}
