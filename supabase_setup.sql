-- =============================================================================
-- TDMU SmartDoc — Supabase SQL Setup
-- Chạy toàn bộ file này 1 lần trong Supabase SQL Editor
-- Tất cả lệnh đều idempotent (an toàn khi chạy lại)
-- =============================================================================

-- 1. Bật pgvector
create extension if not exists vector;

-- 2. Bảng tài liệu (bao gồm status + notebook_id ngay từ đầu)
create table if not exists documents (
  id          uuid        primary key default gen_random_uuid(),
  user_id     text        not null,
  title       text        not null,
  page_count  int         default 0,
  type        text        default 'pdf',
  status      text        default 'processing',
  notebook_id uuid,
  created_at  timestamptz default now()
);

-- Đảm bảo các cột tồn tại cho bảng cũ (idempotent)
alter table documents add column if not exists status      text        default 'processing';
alter table documents add column if not exists notebook_id uuid;

-- 3. Bảng notebooks
create table if not exists notebooks (
  id          uuid        primary key default gen_random_uuid(),
  user_id     text        not null,
  name        text        not null,
  color       text        default '#6750A4',
  icon        text        default 'school',
  summary     text        default '',
  suggestions jsonb       default '[]',
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

-- Đảm bảo cột icon tồn tại cho bảng cũ
alter table notebooks add column if not exists icon text default 'school';

-- 4. Foreign key: documents → notebooks (sau khi notebooks đã tồn tại)
do $$
begin
  if not exists (
    select 1 from information_schema.table_constraints
    where constraint_name = 'documents_notebook_id_fkey'
      and table_name = 'documents'
  ) then
    alter table documents
      add constraint documents_notebook_id_fkey
      foreign key (notebook_id) references notebooks(id) on delete set null;
  end if;
end $$;

-- 5. Bảng chunks + embedding
create table if not exists chunks (
  id          uuid        primary key default gen_random_uuid(),
  document_id uuid        references documents(id) on delete cascade,
  user_id     text        not null,
  content     text        not null,
  page_num    int         default 0,
  embedding   vector(768),
  notebook_id uuid        references notebooks(id) on delete set null,
  created_at  timestamptz default now()
);

-- Đảm bảo cột notebook_id tồn tại cho bảng cũ
alter table chunks add column if not exists notebook_id uuid references notebooks(id) on delete set null;

-- 6. HNSW index (tìm kiếm cosine nhanh)
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);

create index if not exists documents_notebook_idx on documents(notebook_id);
create index if not exists chunks_notebook_idx    on chunks(notebook_id);

-- =============================================================================
-- RPC FUNCTIONS (trả về filename qua JOIN với documents)
-- DROP trước để cho phép thay đổi return type
-- =============================================================================
drop function if exists match_chunks(vector, integer, text);
drop function if exists match_chunks_by_doc(vector, integer, text, uuid);
drop function if exists match_chunks_by_notebook(vector, integer, text, uuid);

-- 7. RPC: tìm chunks gần nhất — toàn bộ tài liệu của user
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
  similarity  float,
  filename    text
)
language sql stable
as $$
  select
    c.id,
    c.document_id,
    c.content,
    c.page_num,
    1 - (c.embedding <=> query_embedding) as similarity,
    coalesce(d.title, '') as filename
  from chunks c
  left join documents d on d.id = c.document_id
  where c.user_id = user_id_filter
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- 8. RPC: tìm chunks theo tài liệu cụ thể
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
  similarity  float,
  filename    text
)
language sql stable
as $$
  select
    c.id,
    c.document_id,
    c.content,
    c.page_num,
    1 - (c.embedding <=> query_embedding) as similarity,
    coalesce(d.title, '') as filename
  from chunks c
  left join documents d on d.id = c.document_id
  where c.user_id = user_id_filter
    and c.document_id = doc_id_filter
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- 9. RPC: tìm chunks theo notebook
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
  similarity  float,
  filename    text
)
language sql stable
as $$
  select
    c.id,
    c.document_id,
    c.content,
    c.page_num,
    1 - (c.embedding <=> query_embedding) as similarity,
    coalesce(d.title, '') as filename
  from chunks c
  left join documents d on d.id = c.document_id
  where c.user_id = user_id_filter
    and c.notebook_id = notebook_id_filter
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
