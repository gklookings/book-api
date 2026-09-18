"""
Apply the Ask Alwaraq migration (creates alwaraq_* tables only).

Usage (from the repo root):
    venv/bin/python -m scripts.alwaraq_setup
"""

from pathlib import Path

from app.langchain.alwaraq_lib import db

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "002_alwaraq_tables.sql"


def main():
    sql = MIGRATION.read_text(encoding="utf-8")
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print(f"Applied {MIGRATION.name}")


if __name__ == "__main__":
    main()
