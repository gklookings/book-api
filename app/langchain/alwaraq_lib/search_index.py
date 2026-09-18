"""
Keyword search index for Ask Alwaraq: `alwaraq_chunk_text`.

A normalized copy of each Alwaraq chunk's text from `books` (read-only source),
with a trigram index so LIKE '%term%' searches are fast across the whole library.

Sync is incremental and safe to run any time:
  - new chunks:     books.id greater than the highest id already copied
                    (/chromadb/upload deletes and re-inserts, so re-uploaded books get new ids)
  - removed chunks: rows whose id no longer exists in books are deleted
"""

import os
import threading
import time

from psycopg2 import errors as pg_errors

from app.langchain.alwaraq_lib import db
from app.langchain.alwaraq_lib.normalize import FOLD_FROM, FOLD_TO, REMOVED_CHARS

BATCH_SIZE = int(os.getenv("ALWARAQ_SYNC_BATCH", "20000"))
AUTO_SYNC_INTERVAL_S = int(os.getenv("ALWARAQ_SYNC_INTERVAL_S", "3600"))
TRGM_INDEX = "idx_alwaraq_chunk_text_trgm"

_sync_lock = threading.Lock()
_state = {"ready": None, "checked_at": 0.0, "last_sync": 0.0, "running": False}
_READY_TTL = 60


def _bookid_pattern() -> str:
    return os.getenv("ALWARAQ_BOOKID_PATTERN", r"^[0-9]+$")


def is_ready(force: bool = False) -> bool:
    """True when the table has rows and its trigram index is valid."""
    now = time.time()
    if not force and _state["ready"] is not None and now - _state["checked_at"] < _READY_TTL:
        return _state["ready"]
    try:
        row = db.fetch_one(
            """
            SELECT EXISTS (SELECT 1 FROM alwaraq_chunk_text LIMIT 1) AS has_rows,
                   EXISTS (
                       SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                       WHERE c.relname = %s AND i.indisvalid
                   ) AS has_index
            """,
            (TRGM_INDEX,),
        )
        ready = bool(row and row["has_rows"] and row["has_index"])
    except pg_errors.UndefinedTable:
        ready = False
    _state.update(ready=ready, checked_at=now)
    return ready


def _run_long(sql: str, params=None) -> int:
    """Statement without the usual timeout (bulk load / index build)."""
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 0")
            cur.execute(sql, params)
            return cur.rowcount


def sync(log=print) -> dict:
    """Copy new chunks, drop removed ones, and make sure the trigram index exists."""
    if not _sync_lock.acquire(blocking=False):
        return {"status": "already_running"}
    _state["running"] = True
    started = time.time()
    try:
        max_copied = db.fetch_one("SELECT COALESCE(MAX(id), 0) AS m FROM alwaraq_chunk_text")["m"]
        max_books = db.fetch_one("SELECT COALESCE(MAX(id), 0) AS m FROM books")["m"]
        inserted = 0
        lo = max_copied
        while lo < max_books:
            hi = lo + BATCH_SIZE
            n = _run_long(
                """
                INSERT INTO alwaraq_chunk_text (id, bookid, text_normalized)
                SELECT id, bookid, translate(text_content, %s, %s)
                FROM books
                WHERE id > %s AND id <= %s AND bookid ~ %s AND text_content IS NOT NULL
                ON CONFLICT (id) DO NOTHING
                """,
                (FOLD_FROM + REMOVED_CHARS, FOLD_TO, lo, hi, _bookid_pattern()),
            )
            inserted += max(n, 0)
            lo = hi
            log(f"[alwaraq] search index: copied up to id {min(hi, max_books)}/{max_books} (+{inserted})")

        deleted = _run_long(
            """
            DELETE FROM alwaraq_chunk_text c
            WHERE NOT EXISTS (SELECT 1 FROM books b WHERE b.id = c.id)
            """
        )

        index_created = False
        if not db.fetch_one(
            "SELECT 1 AS ok FROM pg_class WHERE relname = %s", (TRGM_INDEX,)
        ):
            log("[alwaraq] search index: building trigram index (one-time, may take several minutes)")
            _run_long(
                f"CREATE INDEX IF NOT EXISTS {TRGM_INDEX} "
                "ON alwaraq_chunk_text USING gin (text_normalized gin_trgm_ops)"
            )
            index_created = True
        _run_long("ANALYZE alwaraq_chunk_text")

        _state["last_sync"] = time.time()
        is_ready(force=True)
        result = {
            "status": "done",
            "inserted": inserted,
            "deleted": deleted,
            "index_created": index_created,
            "seconds": int(time.time() - started),
        }
        log(f"[alwaraq] search index sync: {result}")
        return result
    finally:
        _state["running"] = False
        _sync_lock.release()


def maybe_sync_in_background() -> None:
    """Keep the index fresh without blocking requests: at most once per interval."""
    if _state["running"] or time.time() - _state["last_sync"] < AUTO_SYNC_INTERVAL_S:
        return
    if not is_ready():
        return  # initial build is an explicit admin/script action, not automatic
    _state["last_sync"] = time.time()

    def _run():
        try:
            sync()
        except Exception as e:
            print(f"[alwaraq] Background search-index sync failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


def status() -> dict:
    try:
        row = db.fetch_one(
            "SELECT COUNT(*) AS rows, COUNT(DISTINCT bookid) AS books, MAX(id) AS max_id FROM alwaraq_chunk_text"
        )
    except pg_errors.UndefinedTable:
        return {"ready": False, "reason": "table missing: run migrations/003_alwaraq_chunk_text.sql"}
    return {
        "ready": is_ready(force=True),
        "running": _state["running"],
        "rows": row["rows"],
        "books": row["books"],
        "max_id": row["max_id"],
    }
