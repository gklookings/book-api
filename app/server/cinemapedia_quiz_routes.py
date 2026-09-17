import asyncio

from fastapi import APIRouter, HTTPException

from app.langchain import cinemapedia_quiz
from app.langchain.cinemapedia_quiz_generator import (
    CATEGORIES,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    generate_quiz,
)

router = APIRouter(prefix="/cinemapedia-quiz", tags=["cinemapedia-quiz"])


@router.get("/questions")
def get_cinemapedia_quiz(
    qnCount: int,
    category: str = "mix",
    language: str | None = None,
):
    """
    Generate a Cinemapedia multiple-choice quiz.

    Query params:
        qnCount:  any whole number from 1 to 99
        category: films | actors | directors | mix
        language: en | ar (null/empty -> en). "ar" prioritises items with Arabic data.

    Response `en` and `ar` hold the same questions (same order, same option ids).
    Declared sync so FastAPI runs it in its thread pool (DB + embedding work).
    """
    category = (category or "").strip().lower()
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE

    if not MIN_QUESTION_COUNT <= qnCount <= MAX_QUESTION_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"qnCount must be between {MIN_QUESTION_COUNT} and {MAX_QUESTION_COUNT}",
        )
    if category not in CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"category must be one of {list(CATEGORIES)}",
        )
    if language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"language must be one of {list(LANGUAGES)}",
        )

    data, status_code = generate_quiz(qnCount, category, language)
    if status_code != 200:
        return {
            "error": data.get("error", "An error occurred"),
            "status_code": status_code,
        }
    return {**data, "status_code": status_code}


@router.post("/sync")
async def sync_cinemapedia_quiz(startPage: int = 1, endPage: int | None = None):
    """
    Fetch movies from the Cinemapedia API page by page and store them in the
    cinemapedia_quiz vector store. Returns 202 immediately — ingestion runs in a
    background thread. Poll GET /cinemapedia-quiz/sync-status for progress.
    """
    if cinemapedia_quiz.sync_state["running"]:
        return {
            "status": "already_running",
            "sync": cinemapedia_quiz.sync_state,
            "status_code": 409,
        }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, cinemapedia_quiz.sync_movies, startPage, endPage)
    return {
        "status": "processing",
        "message": "Cinemapedia movies are being fetched and stored in the background.",
        "startPage": startPage,
        "endPage": endPage,
        "status_code": 202,
    }


@router.get("/sync-status")
async def cinemapedia_quiz_sync_status():
    return {"sync": cinemapedia_quiz.sync_state, "status_code": 200}


@router.post("/clean")
async def clean_cinemapedia_quiz():
    if cinemapedia_quiz.sync_state["running"]:
        return {"error": "A sync is running; try again later.", "status_code": 409}
    try:
        data, status_code = cinemapedia_quiz.clean_vector_store()
        if status_code == 200:
            return {"status": data["status"], "status_code": status_code}
        return {
            "error": data.get("error", "An error occurred"),
            "status_code": status_code,
        }
    except Exception as e:
        return {"error": str(e), "status_code": 500}
