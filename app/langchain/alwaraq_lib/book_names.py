"""
Fill in missing book names for Ask Alwaraq.

Books without a catalogue row (alwaraq_books) are looked up on the Alwaraq API in
a background thread and saved to the catalogue, so later answers show their name.
Failures (e.g. the API being down) are retried at most once per RETRY_AFTER_S per book.
"""

import os
import queue
import threading
import time

import requests

from app.langchain.alwaraq_lib import db

# Note: adding a `language` parameter makes this endpoint return HTTP 500.
API_URL = "https://alwaraq.net/json_bookallpages.php?bookId={book_id}"
RETRY_AFTER_S = int(os.getenv("ALWARAQ_NAME_RETRY_S", "86400"))
TIMEOUT_S = 120

_queue: "queue.Queue[str]" = queue.Queue()
_attempted: dict[str, float] = {}
_lock = threading.Lock()
_worker: threading.Thread | None = None


def fetch_book_meta(book_id: str) -> dict | None:
    """{"name": ..., "author": ...} from the Alwaraq API, or None."""
    resp = requests.get(API_URL.format(book_id=book_id), timeout=TIMEOUT_S)
    if resp.status_code != 200:
        print(f"[alwaraq] Book name lookup for {book_id}: HTTP {resp.status_code}")
        return None
    data = resp.json()
    if not isinstance(data, dict) or not str(data.get("name") or "").strip():
        return None
    return {"name": str(data["name"]).strip(), "author": str(data.get("author") or "").strip() or None}


def _save(book_id: str, meta: dict) -> None:
    db.execute(
        """
        INSERT INTO alwaraq_books (book_id, legacy_bookid, title_ar, author_en)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (book_id) DO UPDATE SET
            title_ar = EXCLUDED.title_ar,
            author_en = COALESCE(alwaraq_books.author_en, EXCLUDED.author_en)
        """,
        (book_id, book_id, meta["name"], meta["author"]),
    )


def _run(on_saved) -> None:
    while True:
        book_id = _queue.get()
        try:
            meta = fetch_book_meta(book_id)
            if meta:
                _save(book_id, meta)
                on_saved()
                print(f"[alwaraq] Book name saved: {book_id} = {meta['name']}")
        except Exception as e:
            print(f"[alwaraq] Book name lookup for {book_id} failed: {e}")
        finally:
            _queue.task_done()


def request_names(book_ids: list[str], on_saved=lambda: None) -> None:
    """Queue background lookups for books that have no name yet (non-blocking)."""
    global _worker
    now = time.time()
    with _lock:
        todo = [b for b in book_ids if now - _attempted.get(b, 0) > RETRY_AFTER_S]
        for b in todo:
            _attempted[b] = now
            _queue.put(b)
        if todo and (_worker is None or not _worker.is_alive()):
            _worker = threading.Thread(target=_run, args=(on_saved,), daemon=True)
            _worker.start()
