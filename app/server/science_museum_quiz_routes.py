from fastapi import APIRouter, HTTPException

from app.langchain import science_museum_quiz
from app.langchain.science_museum_quiz_generator import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    generate_quiz,
)

router = APIRouter(prefix="/science-museum-quiz", tags=["science-museum-quiz"])


@router.get("/questions")
def get_science_museum_quiz(
    qnCount: int,
    language: str | None = None,
    fieldId: int | None = None,
):
    """
    Generate a Science Museum multiple-choice quiz from a fresh random set of
    scientists fetched live from the Science Museum API (nothing is stored).

    Query params:
        qnCount:  any whole number from 1 to 99
        language: en | ar (null/empty -> en). Scientists are fetched in this
                  language; the other language is a translation.
        fieldId:  optional, only scientists of that field (see GET /science-museum-quiz/fields)

    Response `en` and `ar` hold the same questions (same order, same option ids).
    Declared sync so FastAPI runs it in its thread pool (HTTP + LLM work).
    """
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE

    if not MIN_QUESTION_COUNT <= qnCount <= MAX_QUESTION_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"qnCount must be between {MIN_QUESTION_COUNT} and {MAX_QUESTION_COUNT}",
        )
    if language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"language must be one of {list(LANGUAGES)}",
        )
    if fieldId is not None and fieldId < 1:
        raise HTTPException(status_code=400, detail="fieldId must be a positive integer")

    data, status_code = generate_quiz(qnCount, language, fieldId)
    if status_code != 200:
        return {
            "error": data.get("error", "An error occurred"),
            "status_code": status_code,
        }
    return {**data, "status_code": status_code}


@router.get("/fields")
def get_science_museum_fields():
    """Field ids and names (en + ar) accepted by `fieldId`."""
    fields = science_museum_quiz.get_fields()
    return {
        "fields": [{"fieldId": field_id, **field} for field_id, field in sorted(fields.items())],
        "status_code": 200,
    }
