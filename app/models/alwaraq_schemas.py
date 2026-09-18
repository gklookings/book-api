from typing import Literal

from pydantic import BaseModel, Field


class AlwaraqFeedbackRequest(BaseModel):
    query_id: str
    feedback: Literal[-1, 1]


class AlwaraqBook(BaseModel):
    book_id: str = Field(..., description="Alwaraq bookId")
    legacy_bookid: str | None = Field(
        None, description="books.bookid used by /chromadb/answer (= document_id). Defaults to book_id."
    )
    title_ar: str
    title_en: str | None = None
    author_ar: str | None = None
    author_en: str | None = None
    author_death_ah: int | None = None
    genre: str | None = None
    edition_info: str | None = None
    description: str | None = None
    language: str = "ar"
    is_pilot: bool = False


class AlwaraqRegisterBooksRequest(BaseModel):
    books: list[AlwaraqBook]
    build_profiles: bool = Field(True, description="Build library-search routing profiles for these books")


class AlwaraqBuildProfilesRequest(BaseModel):
    document_ids: list[str] | None = Field(
        None, description="books.bookid values; omit to rebuild every catalogue book"
    )
