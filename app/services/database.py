import os
from supabase import create_client, Client

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL và SUPABASE_SERVICE_KEY chưa được cấu hình")
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Schema SQL cần chạy 1 lần trong Supabase SQL Editor:
# ---------------------------------------------------------------------------
SUPABASE_SCHEMA_SQL = """
-- Bật pgvector
create extension if not exists vector;

-- Bảng lưu thông tin tài liệu
create table if not exists documents (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null,
  title       text not null,
  page_count  int  default 0,
  type        text default 'pdf',
  status      text default 'ready',
  created_at  timestamptz default now()
);

-- Bảng lưu các đoạn văn bản + vector embedding
create table if not exists chunks (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  user_id     text not null,
  content     text not null,
  page_num    int  default 0,
  embedding   vector(3072),
  created_at  timestamptz default now()
);

-- HNSW index để tìm kiếm vector nhanh
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);
"""
