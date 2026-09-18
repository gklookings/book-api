-- Migration: 002_alwaraq_tables.sql
-- Purpose: Tables for Ask Alwaraq (/alwaraq/*).
-- Creates NEW tables only. Does not alter `books` or any existing table.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Catalogue of books (standardised names, edition info, pilot flag)
CREATE TABLE IF NOT EXISTS alwaraq_books (
    book_id          VARCHAR(64) PRIMARY KEY,      -- Alwaraq bookId
    legacy_bookid    VARCHAR(255) UNIQUE,          -- = document_id param = books.bookid
    title_ar         TEXT NOT NULL,
    title_en         TEXT,
    author_ar        TEXT,
    author_en        TEXT,
    author_death_ah  INTEGER,
    genre            VARCHAR(64),
    edition_info     TEXT,
    description      TEXT,                         -- short description, used for routing
    language         VARCHAR(8) DEFAULT 'ar',
    is_pilot         BOOLEAN NOT NULL DEFAULT FALSE,
    page_indexed_at  TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Page-level passages with full citation metadata (filled in Phase 2)
CREATE TABLE IF NOT EXISTS alwaraq_passages (
    id               BIGSERIAL PRIMARY KEY,
    book_id          VARCHAR(64) NOT NULL REFERENCES alwaraq_books(book_id) ON DELETE CASCADE,
    volume           INTEGER,
    page_no          INTEGER NOT NULL,
    chapter          TEXT,
    section          TEXT,
    chunk_index      INTEGER NOT NULL DEFAULT 0,
    text             TEXT NOT NULL,
    text_normalized  TEXT NOT NULL,
    embedding        vector(768) NOT NULL,
    content_type     VARCHAR(16),
    UNIQUE (book_id, volume, page_no, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_alwaraq_passages_book ON alwaraq_passages(book_id, page_no);
CREATE INDEX IF NOT EXISTS idx_alwaraq_passages_trgm
    ON alwaraq_passages USING gin (text_normalized gin_trgm_ops);
-- HNSW needs pgvector >= 0.5.0; don't fail the migration on older versions.
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_alwaraq_passages_embedding
        ON alwaraq_passages USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN others THEN
    RAISE NOTICE 'HNSW index not created (%). Vector search on alwaraq_passages will use a scan.', SQLERRM;
END $$;

-- One routing vector per book, used to pick books when document_id is omitted.
-- Built by READING books (never writing to it).
CREATE TABLE IF NOT EXISTS alwaraq_book_profiles (
    legacy_bookid      VARCHAR(255) PRIMARY KEY,   -- = books.bookid
    profile_text       TEXT,
    profile_embedding  vector(768),
    centroid_embedding vector(768),
    chunk_count        INTEGER,
    built_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- People & places (Phase 3)
CREATE TABLE IF NOT EXISTS alwaraq_entities (
    id               BIGSERIAL PRIMARY KEY,
    type             VARCHAR(16) NOT NULL,         -- person | place
    name_ar          TEXT NOT NULL,
    name_en          TEXT,
    variants         TEXT[] DEFAULT '{}',
    death_ah         INTEGER,
    lat              DOUBLE PRECISION,
    lon              DOUBLE PRECISION,
    modern_name      TEXT,
    description      TEXT
);
CREATE TABLE IF NOT EXISTS alwaraq_entity_mentions (
    entity_id        BIGINT REFERENCES alwaraq_entities(id) ON DELETE CASCADE,
    passage_id       BIGINT REFERENCES alwaraq_passages(id) ON DELETE CASCADE,
    surface_form     TEXT,
    PRIMARY KEY (entity_id, passage_id)
);

-- Research Notebook (Phase 3)
CREATE TABLE IF NOT EXISTS alwaraq_notebooks (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token    VARCHAR(512) NOT NULL,
    title            TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS alwaraq_notebook_items (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notebook_id      UUID NOT NULL REFERENCES alwaraq_notebooks(id) ON DELETE CASCADE,
    passage_id       BIGINT REFERENCES alwaraq_passages(id),
    quote            TEXT NOT NULL,
    citation         JSONB NOT NULL,
    note             TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Query log for evaluation, feedback, cost tracking
CREATE TABLE IF NOT EXISTS alwaraq_query_log (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token    VARCHAR(512),
    endpoint         VARCHAR(64) NOT NULL,
    question         TEXT NOT NULL,
    document_id      VARCHAR(255),                 -- NULL = library scope
    retrieved_ids    JSONB,
    answer           JSONB,
    confidence       VARCHAR(16),
    latency_ms       INTEGER,
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    feedback         SMALLINT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alwaraq_query_log_created ON alwaraq_query_log(created_at DESC);
