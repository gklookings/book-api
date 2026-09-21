"""
Ask Alwaraq — cited research answers over the Alwaraq library.

Independent of chroma_store.py (/chromadb/answer) and every other domain
module. It READS the same `books` vector table, and writes only to its own
`alwaraq_*` tables and to the shared memory tables via app.memory.service.

Pipeline (see implementionPlan.md §5):
  1. understand  – rewrite follow-ups using history, sub-queries, keywords
  2. scope       – one book (document_id) or route to top-N books (library)
  3. retrieve    – hybrid vector + keyword search, RRF fusion
  4. rerank      – LLM relevance grading, per-book cap in library scope
  5. compose     – JSON answer where every claim cites passage ids
  6. verify      – drop unsupported quotes/claims, assign confidence
  7. log/memory  – alwaraq_query_log + session memory (Sessiontoken)
"""

import json
import os
import re
import threading
import time

from psycopg2 import errors as pg_errors

from app.langchain.alwaraq_lib import book_names, db, prompts, retrieval, search_index
from app.langchain.alwaraq_lib.normalize import (
    detect_language,
    literal_terms,
    requested_content_language,
)
from app.langchain.alwaraq_lib.verify import (
    strip_unknown_markers,
    verify_answer,
    verify_composition,
)
from app.memory import repository as memory_repository
from app.memory import service as memory_service

MEMORY_DOMAIN = "alwaraq"

# Questions that ask to be written for, not cited to. These are answered in
# "composed" mode: the library's own books are the material, the piece is ours.
GENERATIVE_INTENTS = {"compose", "reading_plan"}
INTENTS = {"fact", "trace", "compare", "reading_plan", "earliest_use", "define", "compose", "other"}

MAIN_MODEL = os.getenv("ALWARAQ_MAIN_MODEL", "gpt-4o")
FAST_MODEL = os.getenv("ALWARAQ_FAST_MODEL", "gpt-4o-mini")
MAX_PASSAGES = int(os.getenv("ALWARAQ_MAX_PASSAGES", "12"))
ROUTE_TOP_BOOKS = int(os.getenv("ALWARAQ_ROUTE_TOP_BOOKS", "8"))
RERANK_CANDIDATES = int(os.getenv("ALWARAQ_RERANK_CANDIDATES", "30"))
PER_BOOK_CAP = 4
PERMALINK_TEMPLATE = os.getenv("ALWARAQ_PERMALINK_TEMPLATE", "")  # e.g. https://…?bookId={book_id}&page={page}
# How many of the session's previous library answers contribute pinned books.
SESSION_BOOK_LOOKBACK = int(os.getenv("ALWARAQ_SESSION_BOOK_LOOKBACK", "3"))
# How many of them a follow-up question is narrowed to.
FOLLOW_UP_BOOKS_MAX = int(os.getenv("ALWARAQ_FOLLOW_UP_BOOKS", "3"))
# Seconds an answer waits for the names of the books it searched. The upstream
# endpoint builds the whole book before sending a byte (~7s to first byte), so
# waiting is a poor trade and the default is not to: the lookups still start,
# in parallel, and land within seconds — in time for the next question. Fill the
# catalogue ahead of time instead (backfill_book_names) and none of this runs.
NAME_BUDGET_S = float(os.getenv("ALWARAQ_NAME_BUDGET_S", "0"))

_llms: dict = {}
_llm_lock = threading.Lock()


class AlwaraqError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# ── LLM helpers ───────────────────────────────────────────────────────────────


def _get_llm(model: str):
    if model not in _llms:
        with _llm_lock:
            if model not in _llms:
                from langchain_openai import ChatOpenAI

                _llms[model] = ChatOpenAI(model=model, temperature=0).bind(
                    response_format={"type": "json_object"}
                )
    return _llms[model]


def _invoke_json(model: str, prompt: str, usage: dict) -> dict:
    response = _get_llm(model).invoke(prompt)
    meta = getattr(response, "usage_metadata", None) or {}
    usage["tokens_in"] += int(meta.get("input_tokens", 0) or 0)
    usage["tokens_out"] += int(meta.get("output_tokens", 0) or 0)
    content = (response.content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
    return json.loads(content)


# ── Steps ────────────────────────────────────────────────────────────────────


def _understand(
    question: str, history: str, library_scope: bool, usage: dict, open_book_title: str | None = None
) -> dict:
    fallback = {
        "standalone_question": question,
        "language": detect_language(question),
        "intent": "other",
        "entities": [],
        "time_range_ah": None,
        "sub_queries": [question],
        "keywords": [],
        "candidate_books": [],
        "content_language": None,
        "about_open_book": False,
        "about_previous_answer": False,
    }
    prompt = prompts.UNDERSTAND_PROMPT.format(
        history_block=prompts.history_block(history),
        open_book_block=prompts.open_book_block(open_book_title),
        question=question,
        scope_description=(
            "the whole library (books must be chosen)" if library_scope else "a single book chosen by the user"
        ),
        candidate_books_instruction=(
            prompts.CANDIDATE_BOOKS_LIBRARY if library_scope else prompts.CANDIDATE_BOOKS_BOOK
        ),
    )
    try:
        data = _invoke_json(FAST_MODEL, prompt, usage)
    except Exception as e:
        print(f"[alwaraq] Understand step failed, using raw question: {e}")
        return fallback

    out = dict(fallback)
    for key in fallback:
        if data.get(key) not in (None, "", []):
            out[key] = data[key]
    out["standalone_question"] = str(out["standalone_question"]).strip() or question
    out["language"] = out["language"] if out["language"] in ("ar", "en") else fallback["language"]
    out["intent"] = out["intent"] if out["intent"] in INTENTS else "other"
    out["about_open_book"] = bool(out["about_open_book"]) and bool(open_book_title)
    out["about_previous_answer"] = bool(out["about_previous_answer"]) and bool(history)
    out["content_language"] = out["content_language"] if out["content_language"] in ("ar", "en") else None
    for key in ("entities", "sub_queries", "keywords", "candidate_books"):
        out[key] = [str(x) for x in out[key] if str(x).strip()] if isinstance(out[key], list) else []
    return out


def _rerank(question: str, candidates: list[dict], usage: dict) -> list[dict]:
    """Adds 'relevance' (0–3) to each candidate. On failure keeps RRF order with relevance=None."""
    for i, p in enumerate(candidates):
        p["rerank_id"] = f"C{i + 1}"
    prompt = prompts.RERANK_PROMPT.format(
        question=question,
        passages=prompts.format_passages(candidates, label_key="rerank_id", max_chars=1800),
    )
    try:
        scores = _invoke_json(FAST_MODEL, prompt, usage).get("scores") or {}
    except Exception as e:
        print(f"[alwaraq] Rerank failed, keeping fusion order: {e}")
        for p in candidates:
            p["relevance"] = None
        return candidates
    for p in candidates:
        try:
            p["relevance"] = int(scores.get(p["rerank_id"], 0))
        except (TypeError, ValueError):
            p["relevance"] = 0
    return candidates


def select_passages(candidates: list[dict], max_sources: int, per_book_cap: int | None) -> list[dict]:
    """Keep relevance >= 2 (or all, if rerank failed), best first, with an optional per-book cap."""
    if any(p.get("relevance") is not None for p in candidates):
        pool = [p for p in candidates if (p.get("relevance") or 0) >= 2]
        pool.sort(key=lambda p: (-(p.get("relevance") or 0), -p.get("rrf", 0)))
    else:
        pool = list(candidates)
    selected, per_book = [], {}
    for p in pool:
        doc = p["document_id"]
        if per_book_cap and per_book.get(doc, 0) >= per_book_cap:
            continue
        per_book[doc] = per_book.get(doc, 0) + 1
        selected.append(p)
        if len(selected) >= max_sources:
            break
    return selected


def _compose(question: str, history: str, language: str, passages: list[dict], usage: dict) -> dict:
    prompt = prompts.COMPOSE_PROMPT.format(
        answer_language_name=prompts.LANGUAGE_NAMES.get(language, "Arabic"),
        history_block=prompts.history_block(history),
        question=question,
        passages=prompts.format_passages(passages, label_key="label", max_chars=2500),
    )
    return _invoke_json(MAIN_MODEL, prompt, usage)


def _compose_piece(
    question: str, history: str, language: str, passages: list[dict], books: list[dict], usage: dict
) -> dict:
    """Write what the reader asked for, out of the library's own books."""
    prompt = prompts.COMPOSITION_PROMPT.format(
        answer_language_name=prompts.LANGUAGE_NAMES.get(language, "Arabic"),
        history_block=prompts.history_block(history),
        question=question,
        books_block=prompts.books_block(books),
        passages=prompts.format_passages(passages, label_key="label", max_chars=2500),
    )
    return _invoke_json(MAIN_MODEL, prompt, usage)


# ── Formatting ───────────────────────────────────────────────────────────────


def _book_meta(document_id: str, language: str) -> dict:
    info = retrieval.get_book_info(document_id) or {}
    if language == "en":
        title = info.get("title_en") or info.get("title_ar")
        author = info.get("author_en") or info.get("author_ar")
    else:
        title = info.get("title_ar") or info.get("title_en")
        author = info.get("author_ar") or info.get("author_en")
    return {"book_id": info.get("book_id") or document_id, "book": title or document_id, "author": author}


def _book_name(document_id: str, language: str) -> str | None:
    info = retrieval.get_book_info(document_id)
    if not info:
        return None
    if language == "en":
        return info.get("title_en") or info.get("title_ar")
    return info.get("title_ar") or info.get("title_en")


def ensure_book_names(document_ids: list[str], language: str) -> None:
    """
    Start the lookup for any of these books whose name is not in the catalogue.

    They are fetched in parallel and, with the default budget of 0, nothing is
    waited for: this answer still shows the bare ids, the next one shows names.
    Raise ALWARAQ_NAME_BUDGET_S to let the answer wait (about 10s covers a
    library-scope set of 8), or fill the catalogue in advance instead.
    """
    missing = [d for d in dict.fromkeys(document_ids) if not _book_name(d, language)]
    if not missing:
        return
    saved = book_names.ensure_names(missing, NAME_BUDGET_S, retrieval.invalidate_catalogue)
    retrieval.invalidate_catalogue()
    print(f"[alwaraq] Book names: {len(missing)} missing, {saved} resolved within {NAME_BUDGET_S}s")


def _book_refs(document_ids: list[str], language: str) -> list[dict]:
    """[{"bookId", "bookName"}]; a name still missing is looked up in the background."""
    refs, missing = [], []
    for doc_id in document_ids:
        name = _book_name(doc_id, language)
        if not name:
            missing.append(doc_id)
        refs.append({"bookId": doc_id, "bookName": name})
    if missing:
        book_names.request_names(missing, on_saved=retrieval.invalidate_catalogue)
    return refs


def _permalink(book_id: str, page) -> str | None:
    if not PERMALINK_TEMPLATE or page is None:
        return None
    return PERMALINK_TEMPLATE.format(book_id=book_id, page=page)


def _source_entry(p: dict, language: str, quotes: list[dict] | None) -> dict:
    meta = _book_meta(p["document_id"], language)
    first = (quotes or [None])[0] or {}
    return {
        "document_id": p["document_id"],
        "book_id": meta["book_id"],
        "book": meta["book"],
        "author": meta["author"],
        "volume": p.get("volume"),
        "page": p.get("page"),
        "chapter": p.get("chapter"),
        "citation_level": p["citation_level"],
        "quote": first.get("text"),
        "quote_verbatim": first.get("verbatim"),
        "context_before": first.get("context_before", ""),
        "context_after": first.get("context_after", ""),
        "quotes": [q["text"] for q in quotes or []],
        "text": p["text"],
        "passage_id": p.get("passage_id"),
        "permalink": _permalink(meta["book_id"], p.get("page")),
    }


def _compact_for_memory(summary: str, sources: dict) -> str:
    """Short answer + source line saved to memory (not the full JSON)."""
    refs = []
    for s in sources.values():
        ref = s["book"]
        if s.get("page") is not None:
            ref += f" ج{s.get('volume') or '-'} ص{s['page']}"
        refs.append(ref)
    refs = list(dict.fromkeys(refs))
    text = re.sub(r"\s*\[P\d+\]", "", summary or "").strip()
    return f"{text}\nSources: {'; '.join(refs)}" if refs else text


def recent_session_books(session_token: str | None) -> list[str]:
    """
    Books that answered this session's recent library questions.

    Routing otherwise starts from scratch every turn, so a follow-up ("who
    wrote it?") can be routed away from the very book that supplied the quote
    it is asking about. These books are kept in the routed set.
    """
    if not session_token:
        return []
    try:
        rows = db.fetch_all(
            """
            SELECT answer -> 'books_cited' AS books
            FROM alwaraq_query_log
            WHERE session_token = %s AND document_id IS NULL
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_token, SESSION_BOOK_LOOKBACK),
        )
    except Exception as e:  # a cold log table must not stop an answer
        print(f"[alwaraq] Recent-session books unavailable: {e}")
        return []
    out: list[str] = []
    for row in rows:
        for book in row.get("books") or []:
            if isinstance(book, str) and book and book not in out:
                out.append(book)
    return out


# ── Logging ──────────────────────────────────────────────────────────────────


def _log_query(entry: dict) -> str | None:
    try:
        row = db.execute_returning(
            """
            INSERT INTO alwaraq_query_log
                (session_token, endpoint, question, document_id, retrieved_ids, answer,
                 confidence, latency_ms, tokens_in, tokens_out)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                entry.get("session_token"),
                entry["endpoint"],
                entry["question"],
                entry.get("document_id"),
                json.dumps(entry.get("retrieved_ids") or [], ensure_ascii=False),
                json.dumps(entry.get("answer") or {}, ensure_ascii=False, default=str),
                entry.get("confidence"),
                entry.get("latency_ms"),
                entry.get("tokens_in"),
                entry.get("tokens_out"),
            ),
        )
        return str(row["id"]) if row else None
    except Exception as e:
        print(f"[alwaraq] Query log write failed (answer still returned): {e}")
        return None


# ── Public API ───────────────────────────────────────────────────────────────


def answer_question(
    query: str,
    document_id: str | None = None,
    session_token: str | None = None,
    lang: str | None = None,
    max_sources: int | None = None,
    context_book_id: str | None = None,
) -> tuple[dict, int]:
    started = time.time()
    usage = {"tokens_in": 0, "tokens_out": 0}
    timings: dict[str, int] = {}
    mark = [time.time()]

    def _lap(stage: str):
        now = time.time()
        timings[stage] = int((now - mark[0]) * 1000)
        mark[0] = now

    query = (query or "").strip()
    document_id = (document_id or "").strip() or None
    # The book the reader has open, even when they are searching the whole
    # library. Without it "this novel" has no referent at all.
    context_book_id = (context_book_id or "").strip() or None
    max_sources = max(1, min(int(max_sources or MAX_PASSAGES), 20))
    library_scope = document_id is None

    if not query:
        return {"error": "query is required"}, 400

    try:
        # ── Memory: load ──────────────────────────────────────────────────────
        history = ""
        if session_token:
            try:
                history = memory_service.load_context(session_token=session_token, domain=MEMORY_DOMAIN)
            except Exception as e:
                print(f"[alwaraq] Memory load failed, answering statelessly: {e}")

        # ── 1. Understand ─────────────────────────────────────────────────────
        open_book_title = (
            _book_meta(context_book_id, detect_language(query))["book"] if context_book_id else None
        )
        understanding = _understand(query, history, library_scope, usage, open_book_title)
        _lap("understand")
        standalone = understanding["standalone_question"]
        # Detected in code: the LLM is unreliable at reporting the question's language.
        language = lang if lang in ("ar", "en") else detect_language(query)
        # Asking to be written for ("a two-line hook for this novel") is not asking
        # what a text says: composed mode writes from the books instead of citing them.
        mode = "composed" if understanding["intent"] in GENERATIVE_INTENTS else "evidence"
        # "a good book in english" is about the language of the BOOKS, not the answer.
        # Read in code as well: the rewrite step drops it more often than not.
        content_language = understanding["content_language"] or requested_content_language(query)
        # "this novel" means the book on the reader's screen, whatever the scope toggle says.
        scope_book = document_id or (context_book_id if understanding["about_open_book"] else None)
        # A question that follows on from the last answer is about the books that
        # answer came from. Merely nudging routing towards them is not enough: they
        # lose a 30-slot pool to seven other books, which is how a question about
        # the David Copperfield just recommended was answered from Arabic ethics
        # treatises. A follow-up searches those books and no others.
        previous_books = recent_session_books(session_token) if scope_book is None else []
        follow_up_books = (
            previous_books[:FOLLOW_UP_BOOKS_MAX]
            if scope_book is None and understanding["about_previous_answer"]
            else []
        )
        library_scope = scope_book is None and not follow_up_books
        print(f"[alwaraq] Standalone question: {standalone!r} | scope={'library' if library_scope else scope_book}"
              f" | intent={understanding['intent']} | mode={mode}"
              f"{f' | books in {content_language}' if content_language else ''}")

        # ── 2. Scope ─────────────────────────────────────────────────────────
        search_index.maybe_sync_in_background()
        # Titles and names exactly as the reader typed them. The rewrite step
        # translates a title ("Three Essays On America" -> "ثلاث مقالات عن
        # أمريكا") and the one string that occurs verbatim in the text is lost,
        # so it is carried to search and routing separately — and first, since
        # it is the most distinctive term available.
        literals = literal_terms(query)
        if literals:
            print(f"[alwaraq] Literal terms from the question: {literals}")
        if library_scope:
            route_queries = [standalone]
            route_vecs = [db.to_vector_literal(v) for v in retrieval.embed(route_queries)]
            books_searched = retrieval.route_books(
                route_vecs,
                names=understanding["candidate_books"] + understanding["entities"] + literals,
                keywords=understanding["keywords"],
                top_n=ROUTE_TOP_BOOKS,
                entities=understanding["entities"] + literals,
                # the open book first: a library-wide question asked while reading
                # something is usually still partly about what is on the screen
                pinned=([context_book_id] if context_book_id else []) + previous_books,
                content_language=content_language,
            )
            if not books_searched:
                raise AlwaraqError("No books could be selected for this question.", 404)
        elif follow_up_books:
            books_searched = follow_up_books
            print(f"[alwaraq] Following on from the last answer: {books_searched}")
        else:
            if not retrieval.book_exists(scope_book):
                raise AlwaraqError(f"No content found for document_id={scope_book}.", 404)
            books_searched = [scope_book]
        ensure_book_names(books_searched, language)
        _lap("scope")
        print(f"[alwaraq] Books searched: {books_searched}")

        # ── 3. Retrieve ──────────────────────────────────────────────────────
        queries = [standalone] + understanding["sub_queries"]
        if language == "en" or detect_language(query) == "en":
            queries.append(query)
        candidates = retrieval.retrieve(
            books_searched,
            queries,
            literals + understanding["keywords"] + understanding["entities"],
            per_query_limit=40 if not library_scope else 20,
            include_global_passages=library_scope,
            limit=RERANK_CANDIDATES,
        )
        _lap("retrieve")

        # ── 4. Rerank & select ───────────────────────────────────────────────
        selected = []
        if candidates:
            # Relevance grading asks "does this passage answer the question", which
            # nothing does when the question is "write me a hook": in composed mode
            # the best-fused passages are the material, ungraded.
            if mode == "evidence":
                _rerank(standalone, candidates, usage)
            selected = select_passages(candidates, max_sources, PER_BOOK_CAP if library_scope else None)
            selected = retrieval.expand_neighbours(selected)
        _lap("rerank")

        passages_by_label = {}
        for i, p in enumerate(selected):
            p["label"] = f"P{i + 1}"
            p["book_title"] = _book_meta(p["document_id"], language)["book"]
            passages_by_label[p["label"]] = p

        # ── 5. Compose & 6. Verify ───────────────────────────────────────────
        verified = {"sections": [], "disagreements": [], "cited_labels": [], "quote_info": {},
                    "dropped": {}, "has_evidence": False}
        composed = {}
        if selected and mode == "composed":
            books = [_book_meta(doc, language) for doc in dict.fromkeys(p["document_id"] for p in selected)]
            composed = _compose_piece(standalone, history, language, selected, books, usage)
            verified = verify_composition(composed, passages_by_label)
        elif selected:
            composed = _compose(standalone, history, language, selected, usage)
            verified = verify_answer(composed, passages_by_label)
        _lap("compose")

        if verified["has_evidence"]:
            status = "ok"
            known = set(verified["cited_labels"])
            summary = strip_unknown_markers(str(composed.get("summary") or ""), known)
            sources = {
                label: _source_entry(passages_by_label[label], language, verified["quote_info"].get(label))
                for label in verified["cited_labels"]
            }
            confidences = [c["confidence"] for s in verified["sections"] for c in s["claims"]]
            if mode == "composed":
                overall = None  # our own writing, not a graded claim about a text
            else:
                overall = "confirmed" if confidences and all(c == "confirmed" for c in confidences) else (
                    "probable" if "confirmed" in confidences or "probable" in confidences else "uncertain"
                )
            follow_ups = [str(f) for f in (composed.get("follow_ups") or [])][:3]
        else:
            status = "no_evidence"
            messages = prompts.NO_MATERIAL_MESSAGE if mode == "composed" else prompts.NO_EVIDENCE_MESSAGE
            summary = messages.get(language, messages["ar"])
            if composed.get("no_evidence") and composed.get("summary"):
                summary = f"{summary} {strip_unknown_markers(str(composed['summary']), set())}"
            # Show the closest passages found so the reader can judge for themselves.
            closest = selected[:3] or candidates[:3]
            sources = {}
            for i, p in enumerate(closest):
                label = p.get("label") or f"P{i + 1}"
                p["label"] = label
                sources[label] = _source_entry(p, language, None)
            overall = None
            follow_ups = []

        answer = {
            "summary": summary,
            "sections": verified["sections"],
            "disagreements": verified["disagreements"],
            "confidence": overall,
        }
        # Only books that actually answered; "closest passages" from a no-evidence
        # turn must not steer the next turn.
        books_cited = (
            list(dict.fromkeys(s["document_id"] for s in sources.values())) if status == "ok" else []
        )
        latency_ms = int((time.time() - started) * 1000)
        print(f"[alwaraq] status={status} sources={len(sources)} dropped={verified.get('dropped')} "
              f"latency={latency_ms}ms timings={timings} tokens={usage}")

        # ── 7. Log & memory ──────────────────────────────────────────────────
        query_id = _log_query(
            {
                "session_token": session_token,
                "endpoint": "answer",
                "question": query,
                "document_id": scope_book,
                "retrieved_ids": [p["key"] for p in candidates],
                "answer": {
                    "status": status,
                    "answer": answer,
                    "sources": list(sources),
                    "books_cited": books_cited,
                },
                "confidence": overall,
                "latency_ms": latency_ms,
                **usage,
            }
        )

        has_memory = False
        if session_token:
            has_memory = True
            try:
                memory_service.save_exchange(
                    session_token=session_token,
                    domain=MEMORY_DOMAIN,
                    question=query,
                    answer=_compact_for_memory(summary, sources),
                    # The prose of an answer cannot say which books it came from
                    # in a form the next turn can act on. This can.
                    metadata={
                        "query_id": query_id,
                        "status": status,
                        "answer_mode": mode,
                        "document_id": scope_book,
                        "books_cited": books_cited,
                        "sources": list(sources),
                    },
                )
            except Exception as e:
                print(f"[alwaraq] Memory save failed (answer still returned): {e}")

        return {
            "status": status,
            "query_id": query_id,
            "document_id": scope_book,
            "scope": "library" if library_scope else "book",
            "answer_mode": mode,
            "books_searched": _book_refs(books_searched, language),
            "question": query,
            "standalone_question": standalone,
            "language": language,
            "answer": answer,
            "sources": sources,
            "follow_ups": follow_ups,
            "hasMemory": has_memory,
            "timings_ms": {**timings, "total": latency_ms},
        }, 200

    except AlwaraqError as e:
        return {"error": str(e)}, e.status_code
    except Exception as e:
        print(f"[alwaraq] Error answering question: {e}")
        return {"error": str(e)}, 500


# ── Memory / history ─────────────────────────────────────────────────────────


def clear_memory(session_token: str) -> dict:
    return memory_service.clear_session(session_token=session_token, domain=MEMORY_DOMAIN)


def get_history(session_token: str, limit: int = 50, offset: int = 0) -> list[dict]:
    rows = memory_repository.get_chat_history(session_token, MEMORY_DOMAIN, limit=limit, offset=offset)
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "metadata": r.get("metadata") or {},
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]


# ── Feedback ─────────────────────────────────────────────────────────────────


def record_feedback(query_id: str, feedback: int) -> bool:
    try:
        return db.execute(
            "UPDATE alwaraq_query_log SET feedback = %s WHERE id = %s::uuid", (feedback, query_id)
        ) > 0
    except (pg_errors.InvalidTextRepresentation, pg_errors.UndefinedTable):
        return False


# ── Catalogue ────────────────────────────────────────────────────────────────


def list_books(q: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    # A book may hold a catalogue row for its language alone, before anyone has
    # looked its name up. Those are not entries a reader can be shown.
    rows = [r for r in retrieval.get_catalogue() if r.get("title_ar") or r.get("title_en")]
    if q:
        from app.langchain.alwaraq_lib.normalize import normalize_arabic

        needle = normalize_arabic(q).lower()
        rows = [
            r for r in rows
            if needle in normalize_arabic(
                " ".join(str(r.get(f) or "") for f in ("title_ar", "title_en", "author_ar", "author_en"))
            ).lower()
        ]
    rows = sorted(rows, key=lambda r: (r.get("title_ar") or r.get("title_en") or ""))
    out = []
    for r in rows[offset : offset + limit]:
        item = {k: v for k, v in r.items() if k != "page_indexed_at"}
        item["document_id"] = r.get("legacy_bookid")
        item["page_indexed"] = bool(r.get("page_indexed_at"))
        out.append(item)
    return out


def register_books(books: list[dict]) -> int:
    count = 0
    for b in books:
        db.execute(
            """
            INSERT INTO alwaraq_books
                (book_id, legacy_bookid, title_ar, title_en, author_ar, author_en,
                 author_death_ah, genre, edition_info, description, language, is_pilot)
            VALUES (%(book_id)s, %(legacy_bookid)s, %(title_ar)s, %(title_en)s, %(author_ar)s,
                    %(author_en)s, %(author_death_ah)s, %(genre)s, %(edition_info)s,
                    %(description)s, %(language)s, %(is_pilot)s)
            ON CONFLICT (book_id) DO UPDATE SET
                legacy_bookid = EXCLUDED.legacy_bookid,
                title_ar = EXCLUDED.title_ar,
                title_en = EXCLUDED.title_en,
                author_ar = EXCLUDED.author_ar,
                author_en = EXCLUDED.author_en,
                author_death_ah = EXCLUDED.author_death_ah,
                genre = EXCLUDED.genre,
                edition_info = EXCLUDED.edition_info,
                description = EXCLUDED.description,
                language = EXCLUDED.language,
                is_pilot = EXCLUDED.is_pilot
            """,
            b,
        )
        count += 1
    retrieval.invalidate_catalogue()
    return count


def build_profiles(document_ids: list[str] | None = None) -> dict:
    return retrieval.build_profiles(document_ids)


def detect_book_languages(document_ids: list[str] | None = None) -> dict:
    return retrieval.detect_book_languages(document_ids)


def backfill_book_names(limit: int | None = None) -> dict:
    """
    Give every Alwaraq book in `books` its name, author and subject.

    Walks the upstream catalogue listing (20 books a page, pages fetched in
    parallel) and writes the entries whose id we actually hold. Anything the
    listing does not cover falls back to the per-book endpoint, which is far
    slower — `limit` caps how many of those are attempted.

    Reads `books`; writes only to alwaraq_books.
    """
    rows = db.fetch_all("SELECT DISTINCT bookid FROM books")
    all_books = [r["bookid"] for r in rows if retrieval.is_library_book(r["bookid"])]
    result = book_names.sync_catalogue(all_books)
    retrieval.invalidate_catalogue()

    still_missing = [b for b in all_books if not (_book_name(b, "ar") or _book_name(b, "en"))]
    if still_missing:
        attempts = still_missing if limit is None else still_missing[:limit]
        print(f"[alwaraq] {len(still_missing)} book(s) not in the listing; looking up {len(attempts)} one by one")
        result["named_one_by_one"] = book_names.ensure_names(attempts, budget_s=None)
        retrieval.invalidate_catalogue()
    result["still_unnamed"] = sum(
        1 for b in all_books if not (_book_name(b, "ar") or _book_name(b, "en"))
    )
    print(f"[alwaraq] Book name backfill done: {result}")
    return result


# ── Keyword search index ─────────────────────────────────────────────────────


def sync_search_index() -> dict:
    return search_index.sync()


def search_index_status() -> dict:
    return search_index.status()
