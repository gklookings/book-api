"""
Record which language each book in the library is written in.

Roughly half of this library is English, and until this has run nothing knows
which half — so "suggest a good book in english" can only be routed by passage
similarity, which lands on Arabic dictionaries. Reads `books`; writes only
alwaraq_books.language.

Usage (from the repo root):
    venv/bin/python -m scripts.alwaraq_book_languages

Apply migrations/005_book_language.sql first. Safe to re-run; run it again
after uploading new books.
"""

from app.langchain import alwaraq


def main():
    print(alwaraq.detect_book_languages())


if __name__ == "__main__":
    main()
