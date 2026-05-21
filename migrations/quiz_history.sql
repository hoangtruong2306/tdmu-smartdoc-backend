-- =============================================================================
-- MIGRATION: Quiz History Tables
-- Chạy trong Supabase Dashboard → SQL Editor
-- =============================================================================

-- Bảng 1: Mỗi lần làm quiz = 1 session
CREATE TABLE IF NOT EXISTS quiz_sessions (
  id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          TEXT        NOT NULL,
  notebook_id      UUID        REFERENCES notebooks(id) ON DELETE SET NULL,
  notebook_name    TEXT        NOT NULL,
  difficulty       TEXT        NOT NULL CHECK (difficulty IN ('easy','medium','hard')),
  total_questions  INT         NOT NULL,
  correct_count    INT         NOT NULL,
  score_pct        INT         NOT NULL CHECK (score_pct BETWEEN 0 AND 100),
  is_mock          BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Bảng 2: Từng câu hỏi trong session (kèm đáp án người dùng đã chọn)
CREATE TABLE IF NOT EXISTS quiz_questions (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID        NOT NULL REFERENCES quiz_sessions(id) ON DELETE CASCADE,
  question_index  INT         NOT NULL,
  question        TEXT        NOT NULL,
  options         JSONB       NOT NULL,   -- ["A. ...", "B. ...", "C. ...", "D. ..."]
  correct         TEXT        NOT NULL,   -- "A" | "B" | "C" | "D"
  explanation     TEXT        NOT NULL,
  user_answer     TEXT,                   -- NULL nếu bỏ qua không chọn
  is_correct      BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index tăng tốc query lịch sử theo user + notebook
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_user_id
  ON quiz_sessions (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_quiz_sessions_notebook
  ON quiz_sessions (notebook_id, user_id);

CREATE INDEX IF NOT EXISTS idx_quiz_questions_session
  ON quiz_questions (session_id, question_index);

-- Row Level Security (backend dùng service key nên bypass, nhưng tốt để có)
ALTER TABLE quiz_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_questions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_all_quiz_sessions"
  ON quiz_sessions FOR ALL TO service_role USING (true);

CREATE POLICY "service_role_all_quiz_questions"
  ON quiz_questions FOR ALL TO service_role USING (true);
