"""
Give every book in the library its name, author and subject.

Answers name the books they searched, and a book with no catalogue entry shows
a bare id. This walks the upstream catalogue listing (20 books a page, pages
fetched in parallel) and writes the entries whose id we actually hold — about a
minute for the whole library. Books the listing does not cover fall back to the
per-book endpoint, which costs roughly 7 seconds each.

Usage (from the repo root):
    venv/bin/python -m scripts.alwaraq_backfill_names          # everything
    venv/bin/python -m scripts.alwaraq_backfill_names 50       # cap the slow fallback at 50 books

Safe to re-run; run it again after uploading new books. It does not touch
alwaraq_books.language, which comes from the book's own text
(scripts/alwaraq_book_languages.py).
"""

import sys

from app.langchain import alwaraq


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(alwaraq.backfill_book_names(limit))


if __name__ == "__main__":
    main()
