-- Chạy toàn bộ file này 1 lần trong Supabase SQL Editor

-- 1. Bật pgvector
create extension if not exists vector;

-- 2. Bảng tài liệu
create table if not exists documents (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null,
  title       text not null,
  page_count  int  default 0,
  type        text default 'pdf',
  created_at  timestamptz default now()
);

-- 3. Bảng chunks + embedding
create table if not exists chunks (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  user_id     text not null,
  content     text not null,
  page_num    int  default 0,
  embedding   vector(768),
  created_at  timestamptz default now()
);

-- 4. HNSW index (tìm kiếm cosine nhanh)
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);

-- 5. RPC: tìm chunks gần nhất (tất cả tài liệu của user)
create or replace function match_chunks(
  query_embedding vector(768),
  match_count     int,
  user_id_filter  text
)
returns table (
  id          uuid,
  document_id uuid,
  content     text,
  page_num    int,
  similarity  float
)
language sql stable
as $$
  select
    id, document_id, content, page_num,
    1 - (embedding <=> query_embedding) as similarity
  from chunks
  where user_id = user_id_filter
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 6. RPC: tìm chunks theo tài liệu cụ thể
create or replace function match_chunks_by_doc(
  query_embedding vector(768),
  match_count     int,
  user_id_filter  text,
  doc_id_filter   uuid
)
returns table (
  id          uuid,
  document_id uuid,
  content     text,
  page_num    int,
  similarity  float
)
language sql stable
as $$
  select
    id, document_id, content, page_num,
    1 - (embedding <=> query_embedding) as similarity
  from chunks
  where user_id = user_id_filter
    and document_id = doc_id_filter
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ============================================================
-- NOTEBOOKS FEATURE
-- ============================================================

-- 7. Bảng notebooks
create table if not exists notebooks (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null,
  name        text not null,
  color       text default '#6750A4',
  summary     text default '',
  suggestions jsonb default '[]',
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

-- 8. Thêm notebook_id vào documents và chunks
alter table documents add column if not exists notebook_id uuid references notebooks(id) on delete set null;
alter table chunks    add column if not exists notebook_id uuid references notebooks(id) on delete set null;

-- 9. Index nhanh theo notebook
create index if not exists documents_notebook_idx on documents(notebook_id);
create index if not exists chunks_notebook_idx    on chunks(notebook_id);

-- 10. RPC: tìm chunks theo notebook
create or replace function match_chunks_by_notebook(
  query_embedding    vector(768),
  match_count        int,
  user_id_filter     text,
  notebook_id_filter uuid
)
returns table (
  id          uuid,
  document_id uuid,
  content     text,
  page_num    int,
  similarity  float
)
language sql stable
as $$
  select
    id, document_id, content, page_num,
    1 - (embedding <=> query_embedding) as similarity
  from chunks
  where user_id = user_id_filter
    and notebook_id = notebook_id_filter
  order by embedding <=> query_embedding
  limit match_count;
$$;
