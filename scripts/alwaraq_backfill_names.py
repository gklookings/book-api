"""
Fill in the catalogue name of every Alwaraq book that has none.

Answers name the books they searched, and a book with no catalogue entry shows
a bare id until it has been looked up. Run this once and the ids disappear.

Usage (from the repo root):
    venv/bin/python -m scripts.alwaraq_backfill_names          # every missing book
    venv/bin/python -m scripts.alwaraq_backfill_names 50       # the first 50 only

Safe to re-run: books already in the catalogue are skipped, and a book whose
lookup fails is retried on a later run (see ALWARAQ_NAME_RETRY_S).
"""

import sys

from app.langchain import alwaraq


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(alwaraq.backfill_book_names(limit))


if __name__ == "__main__":
    main()
