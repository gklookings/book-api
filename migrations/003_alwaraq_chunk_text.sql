-- Migration: 003_alwaraq_chunk_text.sql
-- Purpose: Fast keyword search for Ask Alwaraq.
-- A NEW table holding a normalized copy of each Alwaraq chunk's text from `books`,
-- with a trigram index. `books` itself is not altered; it is only read to fill this table.
-- Filled and kept in sync by app/langchain/alwaraq_lib/search_index.py
-- (scripts/alwaraq_sync_search.py or POST /alwaraq/admin/search-index/sync).

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS alwaraq_chunk_text (
    id               BIGINT PRIMARY KEY,           -- = books.id
    bookid           TEXT NOT NULL,                -- = books.bookid
    text_normalized  TEXT NOT NULL,                -- diacritics removed, letter forms folded
    synced_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alwaraq_chunk_text_bookid ON alwaraq_chunk_text(bookid);
-- The trigram index (idx_alwaraq_chunk_text_trgm) is created by the sync job
-- after the initial bulk load, which is much faster than indexing row by row.
