"""
Retrieval for Ask Alwaraq.

Sources:
  - `books` (EXISTING, READ-ONLY): chunks used by /chromadb/answer.
    Columns: bookid, text_content, embedding_vector. Book-level citations only.
  - `alwaraq_passages` (NEW): page-level passages for page-indexed books.

Every query against `books` is filtered by a single bookid, exactly like
/chromadb/answer, so we never scan the whole table.
"""

import hashlib
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from psycopg2 import errors as pg_errors

from app.langchain.alwaraq_lib import db, search_index
from app.langchain.alwaraq_lib.normalize import (
    FOLD_FROM,
    FOLD_TO,
    REMOVED_CHARS,
    normalize_arabic,
    spelling_variants,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Must match the model used to build the `books` table (see chroma_store.py).
# Kept as its own constant on purpose: this module does not import chroma_store.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

RRF_K = 60
# Several vector lists (one per sub-query) vs. one keyword list: give the keyword
# list extra weight so exact matches on rare names/places aren't outvoted.
KEYWORD_WEIGHT = 2.5
# Top keyword hits are always given a slot in the candidate pool (exact matches on
# names/places are what the reader asked about), up to a third of the pool.
KEYWORD_RESERVED = 8
SEARCH_CONCURRENCY = int(os.getenv("ALWARAQ_SEARCH_CONCURRENCY", "4"))
# Chunks are cut mid-sentence, so the sentence that names a work and the one
# that names its author often land in different chunks. Selected passages are
# widened with this many characters from the chunk on either side. 0 disables.
NEIGHBOUR_CHARS = int(os.getenv("ALWARAQ_NEIGHBOUR_CHARS", "600"))

_model = None
_model_lock = threading.Lock()


# ── Embeddings ────────────────────────────────────────────────────────────────


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                print(f"[alwaraq] Loading embedding model {EMBEDDING_MODEL}")
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    vectors = _get_model().encode(texts)
    return [[float(x) for x in v] for v in vectors]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_fetch_all(sql: str, params=None) -> list[dict]:
    """fetch_all that treats a missing alwaraq_* table as 'no rows'."""
    try:
        return db.fetch_all(sql, params)
    except pg_errors.UndefinedTable:
        return []


def _like_pattern(keyword: str) -> str:
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def prepare_keywords(keywords: list[str], max_keywords: int = 8) -> list[str]:
    seen, out = set(), []
    for kw in keywords or []:
        norm = normalize_arabic(str(kw))
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max_keywords:
            break
    return out


def keyword_groups(keywords: list[str], max_keywords: int = 8) -> list[list[str]]:
    """
    One group of accepted spellings per keyword; a hit on any spelling in a
    group counts as a hit for that keyword (see normalize.spelling_variants).
    """
    return [spelling_variants(kw) for kw in prepare_keywords(keywords, max_keywords)]


def _group_match_sql(column: str, groups: list[list[str]]) -> tuple[list[str], list[str]]:
    """(one `(col ILIKE .. OR ..)` expression per group, patterns in order)."""
    exprs, patterns = [], []
    for group in groups:
        exprs.append("(" + " OR ".join(f"{column} ILIKE %s" for _ in group) + ")")
        patterns.extend(_like_pattern(v) for v in group)
    return exprs, patterns


def _books_key(document_id: str, text: str) -> str:
    digest = hashlib.md5(f"{document_id}\x00{text}".encode("utf-8")).hexdigest()[:16]
    return f"b:{digest}"


def _books_passage(
    document_id: str, text: str, score: float | None = None, chunk_id: int | None = None
) -> dict:
    return {
        "key": _books_key(document_id, text),
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "source": "books",
        "citation_level": "book",
        "passage_id": None,
        "volume": None,
        "page": None,
        "chapter": None,
        "score": score,
    }


def _passages_passage(row: dict, document_id: str | None = None) -> dict:
    return {
        "key": f"p:{row['id']}",
        "document_id": document_id or row.get("legacy_bookid") or row["book_id"],
        "text": row["text"],
        "source": "passages",
        "citation_level": "page",
        "passage_id": row["id"],
        "volume": row.get("volume"),
        "page": row.get("page_no"),
        "chapter": row.get("chapter"),
        "score": row.get("score"),
    }


# ── Catalogue ─────────────────────────────────────────────────────────────────

_catalogue_cache: dict = {"rows": None, "at": 0.0}
_CATALOGUE_TTL = 300


def get_catalogue(force: bool = False) -> list[dict]:
    now = time.time()
    if force or _catalogue_cache["rows"] is None or now - _catalogue_cache["at"] > _CATALOGUE_TTL:
        _catalogue_cache["rows"] = _safe_fetch_all(
            """
            SELECT book_id, legacy_bookid, title_ar, title_en, author_ar, author_en,
                   author_death_ah, genre, edition_info, description, language,
                   is_pilot, page_indexed_at
            FROM alwaraq_books
            """
        )
        _catalogue_cache["at"] = now
    return _catalogue_cache["rows"]


def invalidate_catalogue():
    _catalogue_cache["rows"] = None


def get_book_info(document_id: str) -> dict | None:
    for row in get_catalogue():
        if row.get("legacy_bookid") == document_id or row.get("book_id") == document_id:
            return row
    return None


def book_exists(document_id: str) -> bool:
    row = db.fetch_one("SELECT 1 AS ok FROM books WHERE bookid = %s LIMIT 1", (document_id,))
    if row:
        return True
    info = get_book_info(document_id)
    return bool(info and info.get("page_indexed_at"))


# ── Search: books table (read-only) ──────────────────────────────────────────


def vector_search_books(
    document_ids: list[str], vector_literals: list[str], per_book_limit: int
) -> list[list[dict]]:
    """
    One ranked list per query vector, covering all `document_ids` in ONE table scan:
    every distance is computed in a single pass over the books' rows (ids only),
    the top `per_book_limit` per book per query are kept, and only those rows'
    text is fetched (via the primary key).
    """
    if not document_ids or not vector_literals:
        return []
    n = len(vector_literals)
    dist_cols = ", ".join(f"embedding_vector <=> %s::vector AS d{i}" for i in range(n))
    dist_names = ", ".join(f"d{i}" for i in range(n))
    rank_cols = ", ".join(f"row_number() OVER (PARTITION BY bookid ORDER BY d{i}) AS r{i}" for i in range(n))
    keep = " OR ".join(f"r.r{i} <= %s" for i in range(n))
    sql = f"""
        WITH d AS (
            SELECT id, bookid, {dist_cols}
            FROM books
            WHERE bookid = ANY(%s)
        ),
        r AS (SELECT id, bookid, {dist_names}, {rank_cols} FROM d)
        SELECT r.*, b.text_content
        FROM r JOIN books b ON b.id = r.id
        WHERE {keep}
    """
    params = list(vector_literals) + [list(document_ids)] + [per_book_limit] * n
    rows = db.fetch_all(sql, params)
    lists = []
    for i in range(n):
        hits = sorted((r for r in rows if r[f"r{i}"] <= per_book_limit), key=lambda r: r[f"d{i}"])
        lists.append(
            [_books_passage(r["bookid"], r["text_content"], 1 - float(r[f"d{i}"]), r["id"]) for r in hits]
        )
    return lists


def keyword_search_books(document_ids: list[str], keywords: list[str], per_book_limit: int) -> list[dict]:
    """
    Keyword search on normalized text for all `document_ids` in one scan, scored
    by IDF within each book: a keyword that appears in most of a book's chunks
    (e.g. the author's own name) counts for little, a rare one (a specific place
    name) counts for a lot. Returns the top `per_book_limit` hits per book.
    """
    if not document_ids or not keywords:
        return []
    groups = keyword_groups(keywords)
    if not groups:
        return []
    if search_index.is_ready():
        return _keyword_search_indexed(document_ids, groups, per_book_limit)
    n = len(groups)
    group_exprs, group_patterns = _group_match_sql("norm", groups)
    match_cols = ", ".join(f"{expr}::int AS m{i}" for i, expr in enumerate(group_exprs))
    df_cols = ", ".join(f"SUM(m{i}) AS d{i}" for i in range(n))
    score_expr = " + ".join(f"f.m{i} * LN((w.n + 1.0) / (w.d{i} + 1.0))" for i in range(n))
    any_expr = " + ".join(f"f.m{i}" for i in range(n))
    sql = f"""
        WITH b AS (
            SELECT id, bookid,
                   translate(text_content, %s, %s) AS norm
            FROM books
            WHERE bookid = ANY(%s)
        ),
        f AS (SELECT id, bookid, {match_cols} FROM b),
        w AS (SELECT bookid, COUNT(*) AS n, {df_cols} FROM f GROUP BY bookid),
        s AS (
            SELECT f.id, f.bookid, ({score_expr}) AS score
            FROM f JOIN w ON w.bookid = f.bookid
            WHERE ({any_expr}) > 0
        ),
        t AS (
            SELECT id, bookid, score,
                   row_number() OVER (PARTITION BY bookid ORDER BY score DESC) AS rn
            FROM s
        )
        SELECT t.id, t.bookid, t.score, bk.text_content
        FROM t JOIN books bk ON bk.id = t.id
        WHERE t.rn <= %s
        ORDER BY t.score DESC
    """
    # translate() folds letter forms and deletes diacritics in one cheap pass:
    # characters in FROM beyond the length of TO are removed.
    params = [FOLD_FROM + REMOVED_CHARS, FOLD_TO, list(document_ids)]
    params += group_patterns
    params += [per_book_limit]
    rows = db.fetch_all(sql, params)
    return [_books_passage(r["bookid"], r["text_content"], float(r["score"]), r["id"]) for r in rows]


def _keyword_search_indexed(
    document_ids: list[str], groups: list[list[str]], per_book_limit: int
) -> list[dict]:
    """Same scoring as keyword_search_books, using the trigram-indexed alwaraq_chunk_text."""
    n = len(groups)
    group_exprs, patterns = _group_match_sql("text_normalized", groups)
    match_cols = ", ".join(f"{expr}::int AS m{i}" for i, expr in enumerate(group_exprs))
    any_like = " OR ".join(group_exprs)
    df_cols = ", ".join(f"SUM(m{i}) AS d{i}" for i in range(n))
    score_expr = " + ".join(f"f.m{i} * LN((w.n + 1.0) / (df.d{i} + 1.0))" for i in range(n))
    sql = f"""
        WITH f AS (
            SELECT id, bookid, {match_cols}
            FROM alwaraq_chunk_text
            WHERE bookid = ANY(%s) AND ({any_like})
        ),
        w AS (
            SELECT bookid, COUNT(*) AS n FROM alwaraq_chunk_text
            WHERE bookid = ANY(%s) GROUP BY bookid
        ),
        df AS (SELECT bookid, {df_cols} FROM f GROUP BY bookid),
        s AS (
            SELECT f.id, f.bookid, ({score_expr}) AS score
            FROM f JOIN w ON w.bookid = f.bookid JOIN df ON df.bookid = f.bookid
        ),
        t AS (
            SELECT id, bookid, score,
                   row_number() OVER (PARTITION BY bookid ORDER BY score DESC) AS rn
            FROM s
        )
        SELECT t.id, t.bookid, t.score, bk.text_content
        FROM t JOIN books bk ON bk.id = t.id
        WHERE t.rn <= %s
        ORDER BY t.score DESC
    """
    # match_cols is in the SELECT list, so its patterns come before the book filter.
    params = patterns + [list(document_ids)] + patterns + [list(document_ids), per_book_limit]
    rows = db.fetch_all(sql, params)
    return [_books_passage(r["bookid"], r["text_content"], float(r["score"]), r["id"]) for r in rows]


def expand_neighbours(passages: list[dict], chars: int | None = None) -> list[dict]:
    """
    Widen each `books` passage with text from the chunks either side of it.

    Chunks are cut by length, not by sense: the sentence naming a work and the
    one naming its author routinely fall in different chunks, which is enough
    for the composer to answer "the passage does not say". The passage keys and
    citations are unchanged — only the text the composer reads gets wider.
    """
    chars = NEIGHBOUR_CHARS if chars is None else chars
    targets = [p for p in passages if p.get("source") == "books" and p.get("chunk_id")]
    if not chars or not targets:
        return passages
    wanted = sorted({n for p in targets for n in (p["chunk_id"] - 1, p["chunk_id"] + 1)})
    try:
        rows = db.fetch_all(
            "SELECT id, bookid, text_content FROM books WHERE id = ANY(%s)", (wanted,)
        )
    except Exception as e:  # context is a nicety; never fail an answer over it
        print(f"[alwaraq] Neighbour lookup failed, using passages as they are: {e}")
        return passages
    by_id = {r["id"]: r for r in rows}
    for p in targets:
        before = by_id.get(p["chunk_id"] - 1)
        after = by_id.get(p["chunk_id"] + 1)
        head = (before["text_content"] or "")[-chars:].strip() if _same_book(before, p) else ""
        tail = (after["text_content"] or "")[:chars].strip() if _same_book(after, p) else ""
        if head or tail:
            p["text"] = " ".join(part for part in (head, p["text"], tail) if part)
            p["neighbour_context"] = True
    return passages


def _same_book(row: dict | None, passage: dict) -> bool:
    return bool(row) and row["bookid"] == passage["document_id"]


# ── Search: alwaraq_passages (page-level) ────────────────────────────────────

_PASSAGE_COLUMNS = """
    p.id, p.book_id, b.legacy_bookid, p.volume, p.page_no, p.chapter, p.text
"""


def vector_search_passages(book_ids: list[str] | None, vector_literal: str, limit: int) -> list[dict]:
    where = "WHERE p.book_id = ANY(%s)" if book_ids else ""
    params = [vector_literal] + ([book_ids] if book_ids else []) + [vector_literal, limit]
    rows = _safe_fetch_all(
        f"""
        SELECT {_PASSAGE_COLUMNS}, 1 - (p.embedding <=> %s::vector) AS score
        FROM alwaraq_passages p
        JOIN alwaraq_books b ON b.book_id = p.book_id
        {where}
        ORDER BY p.embedding <=> %s::vector
        LIMIT %s
        """,
        params,
    )
    return [_passages_passage(r) for r in rows]


def keyword_search_passages(book_ids: list[str] | None, keywords: list[str], limit: int) -> list[dict]:
    if not keywords:
        return []
    groups = keyword_groups(keywords)
    if not groups:
        return []
    group_exprs, patterns = _group_match_sql("p.text_normalized", groups)
    score_expr = " + ".join(f"{expr}::int" for expr in group_exprs)
    any_expr = " OR ".join(group_exprs)
    where_book = "AND p.book_id = ANY(%s)" if book_ids else ""
    params = patterns + patterns + ([book_ids] if book_ids else []) + [limit]
    rows = _safe_fetch_all(
        f"""
        SELECT {_PASSAGE_COLUMNS}, ({score_expr}) AS score
        FROM alwaraq_passages p
        JOIN alwaraq_books b ON b.book_id = p.book_id
        WHERE ({any_expr}) {where_book}
        ORDER BY score DESC
        LIMIT %s
        """,
        params,
    )
    return [_passages_passage(r) for r in rows]


# ── Fusion ───────────────────────────────────────────────────────────────────


def rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K, weights: list[float] | None = None) -> list[dict]:
    """Weighted Reciprocal Rank Fusion. Passages are identified by their 'key'."""
    scores: dict[str, float] = {}
    first_seen: dict[str, dict] = {}
    for idx, ranked in enumerate(ranked_lists):
        weight = weights[idx] if weights else 1.0
        for rank, passage in enumerate(ranked):
            key = passage["key"]
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
            first_seen.setdefault(key, passage)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    out = []
    for key in ordered:
        passage = dict(first_seen[key])
        passage["rrf"] = scores[key]
        out.append(passage)
    return out


# ── Per-book search ──────────────────────────────────────────────────────────


def _search_page_indexed_book(
    document_id: str, book_id: str, vector_literals: list[str], keywords: list[str], limit: int
) -> list[tuple[list[dict], float]]:
    ranked = [(vector_search_passages([book_id], vec, limit), 1.0) for vec in vector_literals]
    ranked.append((keyword_search_passages([book_id], keywords, limit), KEYWORD_WEIGHT))
    for lst, _ in ranked:
        for p in lst:
            p["document_id"] = document_id
    return ranked


def retrieve(
    document_ids: list[str],
    queries: list[str],
    keywords: list[str],
    per_query_limit: int = 40,
    include_global_passages: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """
    Hybrid search over the given books, fused with RRF. `per_query_limit` is per book.
    Books still on the `books` table are searched together (one vector scan + one
    keyword scan in parallel); page-indexed books use `alwaraq_passages`.
    With `limit`, returns at most `limit` passages and guarantees the best
    keyword hits a place among them.
    """
    queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q][:5]
    vectors = [db.to_vector_literal(v) for v in embed(queries)] if queries else []
    kws = prepare_keywords(keywords)

    page_books, plain_books = [], []
    for doc_id in document_ids:
        info = get_book_info(doc_id)
        if info and info.get("page_indexed_at"):
            page_books.append((doc_id, info["book_id"]))
        else:
            plain_books.append(doc_id)

    ranked_lists: list[tuple[list[dict], float]] = []
    errors: list[str] = []

    def _run(label, fn, *args):
        try:
            return fn(*args)
        except Exception as e:  # one failing search must not fail the whole answer
            errors.append(f"{label}: {e}")
            print(f"[alwaraq] Search failed ({label}): {e}")
            return None

    with ThreadPoolExecutor(max_workers=max(2, SEARCH_CONCURRENCY)) as pool:
        vec_f = pool.submit(_run, "vector", vector_search_books, plain_books, vectors, per_query_limit)
        kw_f = pool.submit(_run, "keyword", keyword_search_books, plain_books, kws, per_query_limit)
        page_fs = [
            pool.submit(_run, doc_id, _search_page_indexed_book, doc_id, book_id, vectors, kws, per_query_limit)
            for doc_id, book_id in page_books
        ]
        for lst in vec_f.result() or []:
            ranked_lists.append((lst, 1.0))
        ranked_lists.append((kw_f.result() or [], KEYWORD_WEIGHT))
        for f in page_fs:
            ranked_lists.extend(f.result() or [])

    if include_global_passages:
        for vec in vectors:
            ranked_lists.append((vector_search_passages(None, vec, per_query_limit), 1.0))

    ranked_lists = [(lst, w) for lst, w in ranked_lists if lst]
    if errors and not ranked_lists:
        raise RuntimeError("Search failed: " + "; ".join(errors[:3]))

    fused = rrf_fuse([lst for lst, _ in ranked_lists], weights=[w for _, w in ranked_lists])
    if not limit:
        return fused

    reserved_keys = []
    for lst, weight in ranked_lists:
        if weight == KEYWORD_WEIGHT:
            reserved_keys.extend(p["key"] for p in lst[:KEYWORD_RESERVED])
    return reserve_slots(fused, reserved_keys, limit, max_reserved=max(1, limit // 3))


def reserve_slots(fused: list[dict], reserved_keys: list[str], limit: int, max_reserved: int) -> list[dict]:
    """Top `limit` of `fused`, swapping the tail for reserved passages that didn't make it."""
    head = fused[:limit]
    head_keys = {p["key"] for p in head}
    reserved_set = set(reserved_keys)
    missing = [p for p in fused if p["key"] in reserved_set and p["key"] not in head_keys][:max_reserved]
    if not missing:
        return head
    # `missing` is only non-empty when `head` is full: drop its lowest-ranked
    # non-reserved passages to make room.
    kept = list(head)
    to_remove = len(missing)
    for i in range(len(kept) - 1, -1, -1):
        if to_remove == 0:
            break
        if kept[i]["key"] not in reserved_set:
            kept.pop(i)
            to_remove -= 1
    return kept + missing


# ── Library scope: book routing ──────────────────────────────────────────────


def _match_named_books(names: list[str]) -> list[str]:
    targets = [normalize_arabic(n).lower() for n in names or [] if n and len(n.strip()) > 1]
    if not targets:
        return []
    matches = []
    for row in get_catalogue():
        if not row.get("legacy_bookid"):
            continue
        fields = [row.get(f) or "" for f in ("title_ar", "title_en", "author_ar", "author_en")]
        haystack = " | ".join(normalize_arabic(f).lower() for f in fields)
        if any(t in haystack for t in targets):
            matches.append(row["legacy_bookid"])
    return matches


# Only these books.bookid values are Alwaraq books (excludes e.g. "diaralaqool",
# "IB-AwardsList", which live in the same table for other features).
BOOKID_PATTERN = re.compile(os.getenv("ALWARAQ_BOOKID_PATTERN", r"^[0-9]+$"))
# The existing ivfflat index on books.embedding_vector (lists=100, L2 ops) is used
# read-only for library-wide chunk search. Higher probes = better recall, slower.
IVFFLAT_PROBES = int(os.getenv("ALWARAQ_IVFFLAT_PROBES", "3"))
VECTOR_ROUTE_HITS = 200
KEYWORD_ROUTE_TIMEOUT_MS = int(os.getenv("ALWARAQ_KEYWORD_ROUTE_TIMEOUT_MS", "25000"))
KEYWORD_ROUTE_MAX_TERMS = 2          # full scan of books (no text index)
KEYWORD_ROUTE_MAX_TERMS_INDEXED = 4  # trigram index on alwaraq_chunk_text


def is_library_book(bookid: str) -> bool:
    return bool(bookid) and bool(BOOKID_PATTERN.match(bookid))


def vector_route(vector_literals: list[str]) -> list[str]:
    """
    Books ranked by how many of the library-wide nearest chunks they own.
    Uses L2 distance (<->) because that is what the existing index supports.
    """
    def _nearest(vec: str) -> list[dict]:
        return db.fetch_all(
            "SELECT bookid FROM books ORDER BY embedding_vector <-> %s::vector LIMIT %s",
            (vec, VECTOR_ROUTE_HITS),
            settings={"ivfflat.probes": IVFFLAT_PROBES},
        )

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(vector_literals))) as pool:
        for rows in pool.map(_nearest, vector_literals):
            for r in rows:
                if is_library_book(r["bookid"]):
                    counts[r["bookid"]] = counts.get(r["bookid"], 0) + 1
    return sorted(counts, key=lambda b: counts[b], reverse=True)


def pick_route_terms(terms: list[str], max_terms: int = KEYWORD_ROUTE_MAX_TERMS) -> list[str]:
    """Most specific terms first: multi-word phrases, then longer words."""
    cleaned = []
    for t in terms or []:
        t = " ".join(str(t).split())
        if len(normalize_arabic(t)) >= 3 and t not in cleaned:
            cleaned.append(t)
    cleaned.sort(key=lambda t: (-len(t.split()), -len(t)))
    return cleaned[:max_terms]


def score_keyword_hits(rows: list[dict], n_terms: int, total_books: int) -> list[str]:
    """IDF over books: a term found in few books counts more than one found everywhere."""
    df = [sum(1 for r in rows if r[f"k{i}"]) for i in range(n_terms)]
    scores = {}
    for r in rows:
        score = 0.0
        for i in range(n_terms):
            hits = int(r[f"k{i}"] or 0)
            if hits:
                score += math.log((total_books + 1) / (df[i] + 1)) * math.log(1 + hits)
        scores[r["bookid"]] = score
    return sorted(scores, key=lambda b: scores[b], reverse=True)


def keyword_route(terms: list[str], total_books: int = 2000) -> list[str]:
    """
    Books that literally contain the most specific names/places of the question.
    Scans the whole `books` table, so it runs with a time cap and is skipped on timeout.
    """
    terms = pick_route_terms(
        terms, KEYWORD_ROUTE_MAX_TERMS_INDEXED if search_index.is_ready() else KEYWORD_ROUTE_MAX_TERMS
    )
    if not terms:
        return []
    if search_index.is_ready():
        return _keyword_route_indexed([normalize_arabic(t) for t in terms], total_books)
    table, column = "books", "text_content"
    cols = ", ".join(f"SUM(({column} ILIKE %s)::int) AS k{i}" for i in range(len(terms)))
    where = " OR ".join([f"{column} ILIKE %s"] * len(terms))
    patterns = [_like_pattern(t) for t in terms]
    try:
        rows = db.fetch_all(
            f"SELECT bookid, {cols} FROM {table} WHERE {where} GROUP BY bookid",
            patterns + patterns,
            settings={"statement_timeout": KEYWORD_ROUTE_TIMEOUT_MS},
        )
    except pg_errors.QueryCanceled:
        print(f"[alwaraq] Keyword routing timed out for {terms}; using vector routing only")
        return []
    rows = [r for r in rows if is_library_book(r["bookid"])]
    return score_keyword_hits(rows, len(terms), total_books)


KEYWORD_ROUTE_TERM_CAP = 5000  # a term matching more chunks than this is too common to route on


def _keyword_route_indexed(terms: list[str], total_books: int) -> list[str]:
    """One capped trigram lookup per term, in parallel; over-common terms are dropped."""

    def _hits(term: str) -> dict[str, int] | None:
        variants = spelling_variants(term)
        any_like = " OR ".join(["text_normalized ILIKE %s"] * len(variants))
        rows = db.fetch_all(
            f"""
            SELECT bookid, COUNT(*) AS hits FROM (
                SELECT bookid FROM alwaraq_chunk_text WHERE {any_like} LIMIT %s
            ) s GROUP BY bookid
            """,
            tuple(_like_pattern(v) for v in variants) + (KEYWORD_ROUTE_TERM_CAP + 1,),
            settings={"statement_timeout": KEYWORD_ROUTE_TIMEOUT_MS},
        )
        if sum(int(r["hits"]) for r in rows) > KEYWORD_ROUTE_TERM_CAP:
            print(f"[alwaraq] Keyword routing: '{term}' is too common, ignored")
            return None
        return {r["bookid"]: int(r["hits"]) for r in rows if is_library_book(r["bookid"])}

    try:
        with ThreadPoolExecutor(max_workers=len(terms)) as pool:
            per_term = [h for h in pool.map(_hits, terms) if h]
    except pg_errors.QueryCanceled:
        print(f"[alwaraq] Keyword routing timed out for {terms}; using vector routing only")
        return []
    books = set().union(*per_term) if per_term else set()
    rows = [{"bookid": b, **{f"k{i}": h.get(b, 0) for i, h in enumerate(per_term)}} for b in books]
    return score_keyword_hits(rows, len(per_term), total_books)


def _profile_route(query_vector_literal: str, column: str, limit: int) -> list[str]:
    rows = _safe_fetch_all(
        f"""
        SELECT legacy_bookid FROM alwaraq_book_profiles
        WHERE {column} IS NOT NULL
        ORDER BY {column} <=> %s::vector
        LIMIT %s
        """,
        (query_vector_literal, limit),
    )
    return [r["legacy_bookid"] for r in rows]


PINNED_BOOKS_MAX = int(os.getenv("ALWARAQ_PINNED_BOOKS", "3"))


def route_books(
    vector_literals: list[str],
    names: list[str],
    keywords: list[str],
    top_n: int,
    entities: list[str] | None = None,
    pinned: list[str] | None = None,
    content_language: str | None = None,
) -> list[str]:
    """
    Pick the most relevant books for a library-scope question. Signals:
      - named:    catalogue books whose title/author the question names (always kept)
      - keyword:  books that contain the question's specific names/places
      - vector:   books owning the nearest chunks library-wide (existing ivfflat index)
      - profiles: optional per-book routing profiles (alwaraq_book_profiles)
    No setup is required: keyword + vector routing work on `books` as-is.
    `names` (book titles/authors) are matched against the catalogue only; the
    keyword scan uses `entities` + `keywords`, since titles rarely occur in the text.
    `pinned` books (e.g. the ones the session already cited) are always kept, so a
    follow-up question is not routed away from the book that answered the last one.
    `content_language` ("a good book in english") keeps only books written in it.
    """
    entities = entities or []
    pinned = [b for b in dict.fromkeys(pinned or []) if b][:PINNED_BOOKS_MAX]
    named = _match_named_books(names)
    with ThreadPoolExecutor(max_workers=2) as pool:
        by_vector_f = pool.submit(vector_route, vector_literals)
        by_keyword_f = pool.submit(keyword_route, list(entities) + list(keywords))
        by_profile = _profile_route(vector_literals[0], "profile_embedding", top_n * 3) if vector_literals else []
        by_centroid = _profile_route(vector_literals[0], "centroid_embedding", top_n * 3) if vector_literals else []
        by_vector = by_vector_f.result()
        by_keyword = by_keyword_f.result()
    print(f"[alwaraq] Routing: named={named[:5]} pinned={pinned} "
          f"keyword={by_keyword[:5]} vector={by_vector[:5]}")

    lists, weights = [], []
    for ranked, weight in ((by_keyword, 1.5), (by_vector, 1.0), (by_profile, 0.5), (by_centroid, 0.5)):
        if ranked:
            lists.append([{"key": b} for b in ranked[: top_n * 3]])
            weights.append(weight)
    fused = rrf_fuse(lists, weights=weights)
    first = list(dict.fromkeys(named + pinned))
    ordered = list(dict.fromkeys(first + [p["key"] for p in fused]))
    ordered = filter_by_language(ordered, content_language, top_n)
    return ordered[: max(top_n, len(first))]


# ── Book language (reads `books`, writes alwaraq_books.language only) ────────

LANGUAGE_SAMPLE_CHARS = 500
_LANGUAGE_UPSERT_BATCH = 500


def detect_book_languages(document_ids: list[str] | None = None) -> dict:
    """
    Record which language each book is written in, taken from its own text.

    A reader asking for "a book in english" means the book's text, and about
    half of this library is English — but nothing recorded that, so a
    recommendation was routed by passage similarity alone and landed on Arabic
    dictionaries and Quranic exegesis. Samples five chunks spread through each
    book and takes the majority, which is steadier than reading the first page
    (front matter is often in the other script). A book that is neither Arabic
    nor English — empty in this store, or a Japanese or Chinese text — is left
    with no language rather than a guessed one, which keeps it out of
    recommendations instead of being offered as an English book.
    """
    from app.langchain.alwaraq_lib.normalize import sample_language

    where = "WHERE text_content IS NOT NULL" + (" AND bookid = ANY(%s)" if document_ids else "")
    rows = db.fetch_all(
        f"""
        SELECT bookid, left(text_content, {LANGUAGE_SAMPLE_CHARS}) AS sample FROM (
            SELECT bookid, text_content,
                   row_number() OVER (PARTITION BY bookid ORDER BY id) AS rn,
                   COUNT(*) OVER (PARTITION BY bookid) AS n
            FROM books {where}
        ) s WHERE rn IN (GREATEST(n / 10, 1), GREATEST(n / 4, 1), GREATEST(n / 2, 1),
                         GREATEST(n * 3 / 4, 1), GREATEST(n * 9 / 10, 1))
        """,
        ([document_ids] if document_ids else None),
    )
    votes: dict[str, list[str]] = {}
    for row in rows:
        if not is_library_book(row["bookid"]):
            continue
        votes.setdefault(row["bookid"], [])
        language = sample_language(row["sample"])
        if language:
            votes[row["bookid"]].append(language)
    languages = {b: (max(set(v), key=v.count) if v else None) for b, v in votes.items()}

    pairs = sorted(languages.items())
    for i in range(0, len(pairs), _LANGUAGE_UPSERT_BATCH):
        batch = pairs[i : i + _LANGUAGE_UPSERT_BATCH]
        values = ", ".join(["(%s, %s, %s)"] * len(batch))
        params = [x for book_id, lang in batch for x in (book_id, book_id, lang)]
        db.execute(
            f"""
            INSERT INTO alwaraq_books (book_id, legacy_bookid, language) VALUES {values}
            ON CONFLICT (book_id) DO UPDATE SET language = EXCLUDED.language
            """,
            params,
        )
    invalidate_catalogue()
    counts: dict[str, int] = {}
    for lang in languages.values():
        key = lang or "neither (empty or another language)"
        counts[key] = counts.get(key, 0) + 1
    result = {"books": len(languages), **counts}
    print(f"[alwaraq] Book languages detected: {result}")
    return result


def book_language(document_id: str) -> str | None:
    return (get_book_info(document_id) or {}).get("language")


def books_in_language(language: str, limit: int) -> list[str]:
    """
    Catalogue books written in `language` — the pool a recommendation draws on.

    Books whose name has been looked up come first: a recommendation has to be
    able to say what it is recommending, and a bare id is no use to a reader.
    """
    rows = [r for r in get_catalogue() if r.get("language") == language and r.get("legacy_bookid")]
    named = [r["legacy_bookid"] for r in rows if r.get("title_ar") or r.get("title_en")]
    rest = [r["legacy_bookid"] for r in rows if not (r.get("title_ar") or r.get("title_en"))]
    return (named + rest)[:limit]


def filter_by_language(document_ids: list[str], language: str | None, top_n: int) -> list[str]:
    """
    Keep the books actually written in the language the reader asked for, topping
    up from the catalogue when routing found too few. Falls back to the books as
    routed rather than leaving the reader with nothing.
    """
    if not language:
        return document_ids
    kept = [d for d in document_ids if book_language(d) == language]
    if len(kept) < top_n:
        kept += [b for b in books_in_language(language, top_n * 2) if b not in kept]
    return kept or document_ids


# ── Library scope: building routing profiles (reads `books` only) ────────────


def _profile_text(info: dict | None) -> str:
    if not info:
        return ""
    parts = [
        info.get("title_ar"),
        info.get("title_en"),
        info.get("author_ar"),
        info.get("author_en"),
        info.get("genre"),
        info.get("description"),
    ]
    return " — ".join(p for p in parts if p)


def _centroid(document_id: str) -> tuple[str | None, int]:
    try:
        row = db.fetch_one(
            "SELECT AVG(embedding_vector)::text AS centroid, COUNT(*) AS n FROM books WHERE bookid = %s",
            (document_id,),
        )
        if row and row["n"]:
            return row["centroid"], int(row["n"])
        return None, 0
    except Exception as e:
        # Older pgvector without avg(vector): average in Python.
        print(f"[alwaraq] AVG(vector) unavailable ({e}); averaging in Python for {document_id}")
        rows = db.fetch_all("SELECT embedding_vector FROM books WHERE bookid = %s", (document_id,))
        if not rows:
            return None, 0
        total = [0.0] * EMBEDDING_DIM
        for r in rows:
            for i, x in enumerate(db.parse_vector(r["embedding_vector"])):
                total[i] += x
        return db.to_vector_literal([x / len(rows) for x in total]), len(rows)


def build_profile(document_id: str) -> dict:
    info = get_book_info(document_id)
    text = _profile_text(info)
    profile_vec = db.to_vector_literal(embed([text])[0]) if text else None
    centroid, n = _centroid(document_id)
    if centroid is None and profile_vec is None:
        return {"document_id": document_id, "status": "skipped", "reason": "no chunks in books and no catalogue text"}
    db.execute(
        """
        INSERT INTO alwaraq_book_profiles
            (legacy_bookid, profile_text, profile_embedding, centroid_embedding, chunk_count, built_at)
        VALUES (%s, %s, %s::vector, %s::vector, %s, NOW())
        ON CONFLICT (legacy_bookid) DO UPDATE SET
            profile_text = EXCLUDED.profile_text,
            profile_embedding = EXCLUDED.profile_embedding,
            centroid_embedding = EXCLUDED.centroid_embedding,
            chunk_count = EXCLUDED.chunk_count,
            built_at = NOW()
        """,
        (document_id, text or None, profile_vec, centroid, n),
    )
    return {"document_id": document_id, "status": "built", "chunks": n}


def build_profiles(document_ids: list[str] | None = None) -> dict:
    """Build routing profiles; default = every catalogue book with a legacy_bookid."""
    if not document_ids:
        document_ids = [r["legacy_bookid"] for r in get_catalogue(force=True) if r.get("legacy_bookid")]
    results = []
    for doc_id in document_ids:
        try:
            results.append(build_profile(doc_id))
        except Exception as e:
            results.append({"document_id": doc_id, "status": "error", "error": str(e)})
        print(f"[alwaraq] Profile: {results[-1]}")
    built = sum(1 for r in results if r["status"] == "built")
    print(f"[alwaraq] Routing profiles built: {built}/{len(results)}")
    return {"built": built, "total": len(results), "results": results}
