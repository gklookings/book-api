-- Migration: 005_book_language.sql
-- Purpose: let the catalogue record what language a book is actually written in,
--          so "a good book in english" can be answered from the library's own
--          English books instead of whatever text happens to look similar.
--
-- alwaraq_books.language now means: the language of the BOOK TEXT, detected from
-- books.text_content (see retrieval.detect_book_languages), not of its metadata.

-- A book can be catalogued by id before anyone has looked its name up.
ALTER TABLE alwaraq_books ALTER COLUMN title_ar DROP NOT NULL;

CREATE INDEX IF NOT EXISTS idx_alwaraq_books_language ON alwaraq_books (language);
