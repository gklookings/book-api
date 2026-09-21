"""
Fill in missing book names for Ask Alwaraq.

Books without a catalogue row (alwaraq_books) are looked up on the Alwaraq API
and saved to the catalogue, so answers can show their name instead of a bare id.

Two ways in:
  - ensure_names(): fetch now, in parallel, within a time budget. Used on the
    request path so the first answer about a book already carries its name.
  - request_names(): queue for a background worker, for whatever did not make
    that budget.
Failures (e.g. the API being down) are retried at most once per RETRY_AFTER_S per book.
"""

import json
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import requests

from app.langchain.alwaraq_lib import db
from app.langchain.alwaraq_lib.normalize import detect_language

# Note: adding a `language` parameter makes this endpoint return HTTP 500.
API_URL = "https://alwaraq.net/json_bookallpages.php?bookId={book_id}"
RETRY_AFTER_S = int(os.getenv("ALWARAQ_NAME_RETRY_S", "86400"))
TIMEOUT_S = int(os.getenv("ALWARAQ_NAME_TIMEOUT_S", "120"))
RESOLVE_WORKERS = int(os.getenv("ALWARAQ_NAME_WORKERS", "8"))
# The endpoint answers with the book's entire text — 9.7 MB for one of them —
# and the two fields wanted are in its first bytes. Read the head, then stop.
HEAD_BYTES = 8192
MAX_SCAN_BYTES = 1_000_000

_queue: "queue.Queue[str]" = queue.Queue()
_attempted: dict[str, float] = {}
_lock = threading.Lock()
_worker: threading.Thread | None = None

_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_AUTHOR_RE = re.compile(r'"author"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _unescape(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw


def _meta_from(text: str, complete: bool = False) -> dict | None:
    """
    {"name", "author"} from the text read so far.

    While the stream is still open both fields must be present: the name
    usually arrives first, and returning on it alone would file the book with
    no author. `complete` is for the end of the stream, where the author field
    is genuinely absent rather than merely not here yet.
    """
    name = _NAME_RE.search(text)
    if not name:
        return None
    value = _unescape(name.group(1)).strip()
    if not value:
        return None
    author = _AUTHOR_RE.search(text)
    if author is None and not complete:
        return None
    return {"name": value, "author": (_unescape(author.group(1)).strip() if author else "") or None}


def fetch_book_meta(book_id: str) -> dict | None:
    """{"name": ..., "author": ...} from the Alwaraq API, or None."""
    with requests.get(API_URL.format(book_id=book_id), timeout=TIMEOUT_S, stream=True) as resp:
        if resp.status_code != 200:
            print(f"[alwaraq] Book name lookup for {book_id}: HTTP {resp.status_code}")
            return None
        raw = b""
        for chunk in resp.iter_content(HEAD_BYTES):
            raw += chunk
            # decode the whole buffer each time: a name may straddle two chunks
            meta = _meta_from(raw.decode("utf-8", "replace"))
            if meta:
                return meta
            if len(raw) >= MAX_SCAN_BYTES:
                break
    # the stream ended (or the scan cap was reached): take a name without an author
    return _meta_from(raw.decode("utf-8", "replace"), complete=True)


def _save(book_id: str, meta: dict) -> None:
    # title_ar is NOT NULL, so the name always goes there; a Latin name is also
    # stored as title_en, which is what an English answer looks for first.
    latin = detect_language(meta["name"]) == "en"
    db.execute(
        """
        INSERT INTO alwaraq_books (book_id, legacy_bookid, title_ar, title_en, author_en, language)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (book_id) DO UPDATE SET
            title_ar = EXCLUDED.title_ar,
            title_en = COALESCE(EXCLUDED.title_en, alwaraq_books.title_en),
            author_en = COALESCE(alwaraq_books.author_en, EXCLUDED.author_en),
            language = EXCLUDED.language
        """,
        (
            book_id,
            book_id,
            meta["name"],
            meta["name"] if latin else None,
            meta["author"],
            "en" if latin else "ar",
        ),
    )


def _fetch_and_save(book_id: str, on_saved) -> str | None:
    try:
        meta = fetch_book_meta(book_id)
        if not meta:
            return None
        _save(book_id, meta)
        on_saved()
        print(f"[alwaraq] Book name saved: {book_id} = {meta['name']}")
        return meta["name"]
    except Exception as e:
        print(f"[alwaraq] Book name lookup for {book_id} failed: {e}")
        return None


def _claim(book_ids: list[str]) -> list[str]:
    """Books not already tried within the retry window, marked as tried now."""
    now = time.time()
    with _lock:
        todo = [b for b in dict.fromkeys(book_ids) if b and now - _attempted.get(b, 0) > RETRY_AFTER_S]
        for b in todo:
            _attempted[b] = now
    return todo


def ensure_names(book_ids: list[str], budget_s: float | None, on_saved=lambda: None) -> int:
    """
    Look the names up now, in parallel, giving up after `budget_s` (None waits
    for all of them). Lookups that miss the deadline keep running and land in
    the catalogue for the next question. Returns how many were saved in time.
    """
    todo = _claim(book_ids)
    if not todo:
        return 0
    pool = ThreadPoolExecutor(max_workers=min(len(todo), RESOLVE_WORKERS))
    try:
        futures = [pool.submit(_fetch_and_save, b, on_saved) for b in todo]
        done, pending = wait(futures, timeout=budget_s)
        if pending:
            print(f"[alwaraq] {len(pending)} book name(s) still resolving; they will appear next time")
        return sum(1 for f in done if f.result())
    finally:
        pool.shutdown(wait=False)  # stragglers finish and still save


def _run(on_saved) -> None:
    while True:
        book_id = _queue.get()
        try:
            _fetch_and_save(book_id, on_saved)
        finally:
            _queue.task_done()


def request_names(book_ids: list[str], on_saved=lambda: None) -> None:
    """Queue background lookups for books that have no name yet (non-blocking)."""
    global _worker
    todo = _claim(book_ids)
    if not todo:
        return
    for b in todo:
        _queue.put(b)
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, args=(on_saved,), daemon=True)
            _worker.start()
