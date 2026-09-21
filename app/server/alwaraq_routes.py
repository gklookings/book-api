"""
Ask Alwaraq endpoints (/alwaraq/*).

Independent of /chromadb/* — see implementionPlan.md.
"""

import asyncio

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer

from app.langchain import alwaraq
from app.models.alwaraq_schemas import (
    AlwaraqBuildProfilesRequest,
    AlwaraqFeedbackRequest,
    AlwaraqRegisterBooksRequest,
)
from app.server.auth import verify_token

router = APIRouter(prefix="/alwaraq", tags=["alwaraq"])

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def require_admin(token: str = Depends(_oauth2_scheme)) -> str:
    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username


@router.get("/answer")
async def alwaraq_answer(
    query: str,
    document_id: str | None = None,
    context_book_id: str | None = None,
    lang: str | None = None,
    max_sources: int | None = None,
    sessiontoken: str | None = Header(default=None),
):
    """
    Ask Alwaraq: a cited answer from the library.

    Query params:
        query:        the question (Arabic or English)            [required]
        document_id:  books.bookid to search one book; omit to search the whole library
        context_book_id: books.bookid of the book the reader has open, sent even when
                      searching the whole library. It resolves "this book"/"this novel"
                      (which then narrows the search to it) and is otherwise kept in the
                      routed set. Without it such a question has no referent at all.
        lang:         "ar" | "en" answer language (default: language of the query)
        max_sources:  max passages cited (default 12, max 20)

    Response:
        answer_mode:  "evidence" — every claim cited to a passage (the default), or
                      "composed" — the reader asked to be written for (a hook, a blurb,
                      a recommendation), so the answer is Alwaraq's own writing, made
                      only from the library's books. Label it as such in the UI.

    Headers:
        Sessiontoken: optional. When present, chat history is loaded and used to
                      understand follow-up questions, and this exchange is saved.
                      Clear it with DELETE /alwaraq/memory.
    """
    data, status_code = await run_in_threadpool(
        alwaraq.answer_question,
        query=query,
        document_id=document_id,
        context_book_id=context_book_id,
        session_token=sessiontoken,
        lang=lang,
        max_sources=max_sources,
    )
    if status_code != 200:
        return {"error": data.get("error", "An error occurred"), "status_code": status_code}
    return {**data, "status_code": status_code}


@router.delete("/memory")
async def alwaraq_clear_memory(sessiontoken: str | None = Header(default=None)):
    """Clear Ask Alwaraq chat history for this session (same as DELETE /memory?domain=alwaraq)."""
    if not sessiontoken:
        raise HTTPException(status_code=400, detail="Sessiontoken header is required to clear memory.")
    try:
        result = await run_in_threadpool(alwaraq.clear_memory, sessiontoken)
        return {
            "status": "cleared",
            "session_token": sessiontoken,
            "domain": alwaraq.MEMORY_DOMAIN,
            "memory_deleted": result["memory_deleted"],
            "history_deleted": result["history_deleted"],
        }
    except Exception as e:
        return {"error": str(e), "status_code": 500}


@router.get("/history")
async def alwaraq_history(
    limit: int = 50,
    offset: int = 0,
    sessiontoken: str | None = Header(default=None),
):
    """Past Ask Alwaraq messages for this session, newest first."""
    if not sessiontoken:
        raise HTTPException(status_code=400, detail="Sessiontoken header is required.")
    try:
        messages = await run_in_threadpool(
            alwaraq.get_history, sessiontoken, max(1, min(limit, 200)), max(0, offset)
        )
        return {"messages": messages, "domain": alwaraq.MEMORY_DOMAIN, "status_code": 200}
    except Exception as e:
        return {"error": str(e), "status_code": 500}


@router.get("/books")
async def alwaraq_books(q: str | None = None, limit: int = 50, offset: int = 0):
    """Catalogue of registered books. document_id in each item is the value to pass to /alwaraq/answer."""
    try:
        books = await run_in_threadpool(alwaraq.list_books, q, max(1, min(limit, 500)), max(0, offset))
        return {"books": books, "count": len(books), "status_code": 200}
    except Exception as e:
        return {"error": str(e), "status_code": 500}


@router.post("/feedback")
async def alwaraq_feedback(request: AlwaraqFeedbackRequest):
    """Thumbs up (1) / down (-1) for an answer, using query_id from /alwaraq/answer."""
    ok = await run_in_threadpool(alwaraq.record_feedback, request.query_id, request.feedback)
    if not ok:
        return {"error": "Unknown query_id", "status_code": 404}
    return {"status": "saved", "status_code": 200}


# ── Admin (JWT from POST /token) ─────────────────────────────────────────────


@router.post("/admin/books")
async def alwaraq_register_books(
    request: AlwaraqRegisterBooksRequest,
    _: str = Depends(require_admin),
):
    """
    Register or update catalogue entries (optional). Library-wide search works
    without this; registering adds book titles/authors to answers and lets
    questions that name a book or author route straight to it.
    """
    books = []
    for b in request.books:
        row = b.model_dump()
        row["legacy_bookid"] = row["legacy_bookid"] or row["book_id"]
        books.append(row)
    try:
        count = await run_in_threadpool(alwaraq.register_books, books)
    except Exception as e:
        return {"error": str(e), "status_code": 500}

    if request.build_profiles:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, alwaraq.build_profiles, [b["legacy_bookid"] for b in books])

    return {
        "status": "registered",
        "count": count,
        "profiles": "building in background" if request.build_profiles else "skipped",
        "status_code": 200,
    }


@router.post("/admin/books/names")
async def alwaraq_backfill_book_names(
    limit: int | None = None,
    _: str = Depends(require_admin),
):
    """
    Give every Alwaraq book its name, author and subject, in the background.

    Answers name the books they searched; a book with no catalogue entry shows a
    bare id. Walks the upstream catalogue listing (20 books a page, in parallel)
    and writes the ids this library holds — about a minute for the whole library.
    `limit` caps the slow per-book fallback used for books the listing omits.
    Reads `books`; never writes to it.
    """
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, alwaraq.backfill_book_names, limit)
    return {
        "status": "processing",
        "message": "Book names are being filled in. Check the server console for the summary.",
        "status_code": 202,
    }


@router.post("/admin/books/languages")
async def alwaraq_detect_book_languages(_: str = Depends(require_admin)):
    """
    Record which language each book is written in, in the background.

    Roughly half this library is English, and until this has run nothing knows
    which half — so "suggest a good book in english" is routed by passage
    similarity alone. Reads `books`; writes only alwaraq_books.language.
    Re-run after uploading new books.
    """
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, alwaraq.detect_book_languages, None)
    return {
        "status": "processing",
        "message": "Book languages are being detected. Check the server console for the summary.",
        "status_code": 202,
    }


@router.post("/admin/profiles/build")
async def alwaraq_build_profiles(
    request: AlwaraqBuildProfilesRequest,
    _: str = Depends(require_admin),
):
    """(Re)build library-search routing profiles in the background. Reads `books`; never writes to it."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, alwaraq.build_profiles, request.document_ids)
    return {
        "status": "processing",
        "message": "Routing profiles are being built in the background. Check the server console for the summary.",
        "status_code": 202,
    }


@router.post("/admin/search-index/sync")
async def alwaraq_sync_search_index(_: str = Depends(require_admin)):
    """
    Build / refresh the keyword search index (alwaraq_chunk_text) in the background.
    First run copies every Alwaraq chunk and builds the trigram index (several minutes);
    later runs only copy new chunks and drop removed ones. Reads `books`; never writes to it.
    """
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, alwaraq.sync_search_index)
    return {
        "status": "processing",
        "message": "Search index sync started. Check GET /alwaraq/admin/search-index/status.",
        "status_code": 202,
    }


@router.get("/admin/search-index/status")
async def alwaraq_search_index_status(_: str = Depends(require_admin)):
    try:
        return {**(await run_in_threadpool(alwaraq.search_index_status)), "status_code": 200}
    except Exception as e:
        return {"error": str(e), "status_code": 500}
