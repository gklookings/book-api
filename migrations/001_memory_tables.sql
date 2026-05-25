-- Migration: 001_memory_tables.sql
-- Purpose: Create persistent user memory + full chat history tables

-- Rolling memory window per session + domain
CREATE TABLE IF NOT EXISTS user_memory (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token    VARCHAR(512) NOT NULL,
    domain           VARCHAR(100) NOT NULL DEFAULT 'general',
    summary          TEXT,                         -- Compressed older context
    recent_messages  JSONB NOT NULL DEFAULT '[]',  -- Max 10 messages
    message_count    INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_token, domain)
);

-- Full append-only chat history log
CREATE TABLE IF NOT EXISTS chat_history (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token    VARCHAR(512) NOT NULL,
    domain           VARCHAR(100) NOT NULL DEFAULT 'general',
    role             VARCHAR(20) NOT NULL,    -- 'user' | 'assistant'
    content          TEXT NOT NULL,
    metadata         JSONB DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_memory_token ON user_memory(session_token);
CREATE INDEX IF NOT EXISTS idx_user_memory_token_domain ON user_memory(session_token, domain);
CREATE INDEX IF NOT EXISTS idx_chat_history_token_domain ON chat_history(session_token, domain);
CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history(session_token, created_at DESC);
