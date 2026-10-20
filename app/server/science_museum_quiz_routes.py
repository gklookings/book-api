from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.langchain import science_museum_quiz
from app.langchain.science_museum_quiz_generator import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    generate_quiz,
)

router = APIRouter(prefix="/science-museum-quiz", tags=["science-museum-quiz"])


def _error(message: str, status_code: int) -> JSONResponse:
    """Every error is a real HTTP status with the same minimal body."""
    return JSONResponse(status_code=status_code, content={"error": message, "status_code": status_code})


@router.get("/questions")
def get_science_museum_quiz(
    qnCount: int,
    language: str | None = None,
    fieldId: str | None = None,
    isNobel: str | None = None,
):
    """
    Generate a Science Museum multiple-choice quiz from a fresh random set of
    scientists fetched live from the Science Museum API (nothing is stored).

    Query params:
        qnCount:  any whole number from 1 to 99
        language: en | ar (null/empty -> en). Scientists are fetched in this
                  language; the other language is a translation.
        fieldId:  optional (null/empty/missing/0 -> all fields), only scientists of
                  that field (see GET /science-museum-quiz/fields)
        isNobel:  optional, 0 | 1 (null/empty/missing -> 0). 1 -> only Nobel Prize winners
                  and only Nobel questions (can be combined with fieldId).

    Response `en` and `ar` hold the same questions (same order, same option ids).
    Errors use the HTTP status and body {"error": message, "status_code": status}:
        400 invalid params, 404 no scientists for the filter, 422 not enough data
        for qnCount questions, 502 Science Museum API down, 500 anything else.
    Declared sync so FastAPI runs it in its thread pool (HTTP + LLM work).
    """
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE

    if not MIN_QUESTION_COUNT <= qnCount <= MAX_QUESTION_COUNT:
        return _error(f"qnCount must be between {MIN_QUESTION_COUNT} and {MAX_QUESTION_COUNT}", 400)
    if language not in LANGUAGES:
        return _error(f"language must be one of {list(LANGUAGES)}", 400)
    fieldId = (fieldId or "").strip()
    if fieldId and not fieldId.isdigit():
        return _error("fieldId must be a positive integer", 400)
    fieldId = int(fieldId or 0) or None
    isNobel = (isNobel or "0").strip() or "0"
    if isNobel not in ("0", "1"):
        return _error("isNobel must be 0 or 1", 400)

    data, status_code = generate_quiz(qnCount, language, fieldId, isNobel == "1")
    if status_code != 200:
        return _error(data.get("error", "An error occurred"), status_code)
    return {**data, "status_code": status_code}


@router.get("/fields")
def get_science_museum_fields():
    """Field ids and names (en + ar) accepted by `fieldId`."""
    fields = science_museum_quiz.get_fields()
    return {
        "fields": [{"fieldId": field_id, **field} for field_id, field in sorted(fields.items())],
        "status_code": 200,
    }
