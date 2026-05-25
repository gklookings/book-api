"""
app/memory/repository.py

Low-level database CRUD for user_memory and chat_history tables.
Uses raw psycopg2 (matching the existing project pattern).
"""

import json
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# ── Connection helper ────────────────────────────────────────────────────────

_DB_URL = os.getenv(
    "POSTGRES_MEMORY_URL",
    "postgresql://aibook:evaibooks_12@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books",
)


def _get_conn():
    """Open a new psycopg2 connection. Caller must close it."""
    return psycopg2.connect(_DB_URL)


# ── user_memory ──────────────────────────────────────────────────────────────


def get_memory(session_token: str, domain: str) -> Optional[dict]:
    """
    Load the user_memory row for (session_token, domain).
    Returns a dict or None if not found.
    """
    sql = """
        SELECT id, session_token, domain, summary, recent_messages,
               message_count, created_at, updated_at
        FROM user_memory
        WHERE session_token = %s AND domain = %s
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_token, domain))
            row = cur.fetchone()
            if row is None:
                return None
            row = dict(row)
            # psycopg2 returns JSONB as str in some configs; normalise to list
            msgs = row["recent_messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            row["recent_messages"] = msgs
            return row
    finally:
        conn.close()


def upsert_memory(
    session_token: str,
    domain: str,
    recent_messages: list,
    message_count: int,
    summary: Optional[str] = None,
) -> None:
    """
    INSERT or UPDATE the user_memory row for (session_token, domain).
    """
    sql = """
        INSERT INTO user_memory
            (session_token, domain, summary, recent_messages, message_count, updated_at)
        VALUES
            (%s, %s, %s, %s::jsonb, %s, NOW())
        ON CONFLICT (session_token, domain)
        DO UPDATE SET
            summary          = EXCLUDED.summary,
            recent_messages  = EXCLUDED.recent_messages,
            message_count    = EXCLUDED.message_count,
            updated_at       = NOW()
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    session_token,
                    domain,
                    summary,
                    json.dumps(recent_messages, ensure_ascii=False),
                    message_count,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def delete_memory(session_token: str, domain: str) -> int:
    """Delete user_memory row. Returns number of rows deleted."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_memory WHERE session_token = %s AND domain = %s",
                (session_token, domain),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


# ── chat_history ─────────────────────────────────────────────────────────────


def append_chat_history(
    session_token: str,
    domain: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
) -> None:
    """Append a single message to the full chat_history log."""
    sql = """
        INSERT INTO chat_history (session_token, domain, role, content, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    session_token,
                    domain,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_chat_history(
    session_token: str,
    domain: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return paginated chat history rows, newest-first."""
    sql = """
        SELECT id, session_token, domain, role, content, metadata, created_at
        FROM chat_history
        WHERE session_token = %s AND domain = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_token, domain, limit, offset))
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_chat_history(session_token: str, domain: str) -> int:
    """Delete all chat_history rows for a session+domain. Returns count."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_history WHERE session_token = %s AND domain = %s",
                (session_token, domain),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()
