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
