"""
Create (if needed) and sync the Ask Alwaraq keyword search index (alwaraq_chunk_text).

Usage (from the repo root):
    venv/bin/python -m scripts.alwaraq_sync_search
Safe to re-run: later runs only copy new chunks and remove deleted ones.
"""

from pathlib import Path

from app.langchain.alwaraq_lib import db, search_index

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "003_alwaraq_chunk_text.sql"


def main():
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATION.read_text(encoding="utf-8"))
    print(f"Applied {MIGRATION.name}")
    print(search_index.sync())
    print(search_index.status())


if __name__ == "__main__":
    main()
