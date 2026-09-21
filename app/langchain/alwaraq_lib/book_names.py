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

# The whole catalogue, 20 books to a page: bookid, name, author and subject.
# One page costs about as much as a single book lookup, so this is the way to
# name the library; json_bookallpages is the per-book fallback.
LIST_URL = "https://alwaraq.net/json_booklist.php"
LIST_PAGE_SIZE = 20  # fixed upstream: limit/pagesize/perpage are all ignored
LIST_WORKERS = int(os.getenv("ALWARAQ_BOOKLIST_WORKERS", "8"))
LIST_MAX_PAGES = int(os.getenv("ALWARAQ_BOOKLIST_MAX_PAGES", "400"))

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


# ── The whole catalogue in one walk ─────────────────────────────────────────


def _entry(row: dict) -> dict | None:
    book_id = str(row.get("bookid") or "").strip()
    name = str(row.get("name") or "").strip()
    if not book_id or not name:
        return None
    return {
        "book_id": book_id,
        "name": name,
        "author": str(row.get("author") or "").strip() or None,
        "genre": str(row.get("subjectName") or "").strip() or None,
    }


def fetch_book_list_page(page: int) -> tuple[list[dict], bool, int]:
    """(books on this page, is this the last page, total books upstream)."""
    resp = requests.get(LIST_URL, params={"page": page}, timeout=TIMEOUT_S)
    if resp.status_code != 200:
        print(f"[alwaraq] Book list page {page}: HTTP {resp.status_code}")
        return [], False, 0
    data = resp.json()
    books = [e for e in (_entry(r) for r in data.get("books") or []) if e]
    return books, bool(data.get("isLastPage")), int(data.get("total") or 0)


def _safe_page(page: int) -> list[dict]:
    try:
        return fetch_book_list_page(page)[0]
    except Exception as e:
        print(f"[alwaraq] Book list page {page} failed: {e}")
        return []


def sync_catalogue(known_ids: list[str]) -> dict:
    """
    Name the books this library holds from the upstream catalogue listing.

    Only ids present in our own store are written: the listing covers books we
    do not have. Upstream is taken as the source of truth for a book's name,
    author and subject; `language` is left alone, because it is detected from
    the book's own text (retrieval.detect_book_languages) and that is the
    question a reader is really asking when they want "a book in english".
    """
    wanted = {str(b) for b in known_ids if b}
    books, is_last, total = fetch_book_list_page(1)
    collected = {b["book_id"]: b for b in books}
    pages = min(max(1, -(-total // LIST_PAGE_SIZE)), LIST_MAX_PAGES) if total else 1
    print(f"[alwaraq] Book list: {total} books upstream, {pages} page(s)")
    if not is_last and pages > 1:
        with ThreadPoolExecutor(max_workers=LIST_WORKERS) as pool:
            for page_books in pool.map(_safe_page, range(2, pages + 1)):
                for b in page_books:
                    collected.setdefault(b["book_id"], b)
    matched = [b for book_id, b in collected.items() if book_id in wanted]
    saved = save_entries(matched)
    result = {
        "upstream": len(collected),
        "in_library": len(wanted),
        "named": saved,
        "not_listed": len(wanted - {b["book_id"] for b in matched}),
    }
    print(f"[alwaraq] Book list sync: {result}")
    return result


def _titles(name: str, author: str | None) -> tuple:
    """Put a name in the column that matches its script; the other stays empty."""
    latin_title = detect_language(name) == "en"
    latin_author = bool(author) and detect_language(author) == "en"
    return (
        None if latin_title else name,
        name if latin_title else None,
        None if latin_author else author,
        author if latin_author else None,
    )


def save_entries(entries: list[dict], batch_size: int = 500) -> int:
    """Upsert names, authors and subjects. Leaves language and everything else."""
    rows = []
    for e in entries:
        title_ar, title_en, author_ar, author_en = _titles(e["name"], e.get("author"))
        rows.append((e["book_id"], e["book_id"], title_ar, title_en, author_ar, author_en, e.get("genre")))
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        values = ", ".join(["(%s, %s, %s, %s, %s, %s, %s)"] * len(batch))
        db.execute(
            f"""
            INSERT INTO alwaraq_books
                (book_id, legacy_bookid, title_ar, title_en, author_ar, author_en, genre)
            VALUES {values}
            ON CONFLICT (book_id) DO UPDATE SET
                title_ar = EXCLUDED.title_ar,
                title_en = EXCLUDED.title_en,
                author_ar = EXCLUDED.author_ar,
                author_en = EXCLUDED.author_en,
                genre = COALESCE(EXCLUDED.genre, alwaraq_books.genre)
            """,
            [x for row in batch for x in row],
        )
    return len(rows)


# ── One book at a time (fallback for anything the listing does not cover) ────


def _save(book_id: str, meta: dict) -> None:
    save_entries([{"book_id": book_id, "name": meta["name"], "author": meta["author"], "genre": None}])


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
