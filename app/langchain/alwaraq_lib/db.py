"""
Database access for Ask Alwaraq.

Own connection pool: does not share connections with any other module.
The existing `books` table (used by /chromadb/answer) is only ever READ here.
"""

import os
import threading
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_HOST = "ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com"
_STATEMENT_TIMEOUT_MS = int(os.getenv("ALWARAQ_STATEMENT_TIMEOUT_MS", "30000"))

_pool = None
_pool_lock = threading.Lock()


def _connect_kwargs() -> dict:
    options = f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}"
    url = os.getenv("ALWARAQ_DB_URL")
    if url:
        return {"dsn": url, "options": options}
    return {
        "host": os.getenv("ALWARAQ_DB_HOST", _DEFAULT_HOST),
        "database": os.getenv("ALWARAQ_DB_NAME", "books"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "connect_timeout": 30,
        "options": options,
    }


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                max_conn = int(os.getenv("ALWARAQ_DB_POOL_MAX", "8"))
                _pool = pool.ThreadedConnectionPool(1, max_conn, **_connect_kwargs())
    return _pool


@contextmanager
def get_conn():
    """Borrow a pooled connection. Commits on success, rolls back on error."""
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


_ALLOWED_SETTINGS = {"statement_timeout", "ivfflat.probes"}


def fetch_all(sql: str, params=None, settings: dict | None = None) -> list[dict]:
    """`settings` are applied with SET LOCAL (this transaction only), e.g. {"ivfflat.probes": 3}."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for name, value in (settings or {}).items():
                if name not in _ALLOWED_SETTINGS:
                    raise ValueError(f"Setting not allowed: {name}")
                cur.execute(f"SET LOCAL {name} = {int(value)}")
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def fetch_one(sql: str, params=None) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params=None) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount


def execute_returning(sql: str, params=None) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


def to_vector_literal(embedding) -> str:
    """Format an embedding as a pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.7f}" for x in embedding) + "]"


def parse_vector(value) -> list[float] | None:
    """pgvector columns come back as '[...]' strings from psycopg2."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip().lstrip("[").rstrip("]")
    return [float(x) for x in text.split(",")] if text else []
