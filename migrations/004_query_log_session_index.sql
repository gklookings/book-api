-- Migration: 004_query_log_session_index.sql
-- Purpose: let a follow-up question look up the books that answered the
--          session's previous questions (alwaraq.recent_session_books).
--          Without this index that lookup is a full scan of the query log.

CREATE INDEX IF NOT EXISTS idx_alwaraq_query_log_session
    ON alwaraq_query_log (session_token, created_at DESC);
