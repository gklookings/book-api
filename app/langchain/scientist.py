from pydantic import BaseModel
from openai import OpenAI
from sqlalchemy import create_engine, text
from app.langchain.scientists_list import SCIENTISTS
import random
import json
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books"

engine = create_engine(DATABASE_URL)
client = OpenAI()

SYSTEM_PROMPT = """
You run a yes/no guessing game.
You MUST answer ONLY Yes or No.
Do NOT reveal the scientist.
"""


class AskQuestion(BaseModel):
    session_id: str
    question: str


class Guess(BaseModel):
    session_id: str
    guess: str


def get_session(session_id):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM game_sessions WHERE session_id = :id"),
            {"id": session_id},
        ).fetchone()
        if result:
            return dict(result._mapping)
        return None


def save_session(session):
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO game_sessions (session_id, scientist, question_count, questions)
                VALUES (:session_id, :scientist, :question_count, :questions)
                ON CONFLICT (session_id)
                DO UPDATE SET
                    scientist = EXCLUDED.scientist,
                    question_count = EXCLUDED.question_count,
                    questions = EXCLUDED.questions
            """
            ),
            {
                "session_id": session["session_id"],
                "scientist": session["scientist"],
                "question_count": session["question_count"],
                "questions": json.dumps(session["questions"]),
            },
        )
        conn.commit()


def start_game():
    secret_scientist = random.choice(SCIENTISTS)
    session_id = str(uuid.uuid4())
    try:
        existing_session = get_session(session_id)
        if existing_session:
            session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "scientist": secret_scientist,
            "question_count": 0,
            "questions": [],
        }

        save_session(session)

        return {
            "message": "Game started",
            "questions_left": 10,
            "secret_chosen": True,
            "session_id": session_id,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


def ask_question(data: AskQuestion):
    try:
        session = get_session(data.session_id)
        print(f"Session: {session}")

        if not session:
            return {"error": "Invalid session"}, 404

        if session["question_count"] >= 10:
            return {"error": "No more questions allowed"}, 400

        scientist = session["scientist"]

        # Ask OpenAI strictly Yes/No
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
              Secret Scientist: {scientist}
              Question: {data.question}
              Answer ONLY Yes or No.
              """,
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4.1", messages=messages, temperature=0
        )

        answer = response.choices[0].message.content.strip()

        # sanitize
        if answer.lower() not in ["yes", "no"]:
            answer = "Yes" if "yes" in answer.lower() else "No"

        # update session
        questions = session["questions"] or []
        questions.append({"q": data.question, "a": answer})

        session["question_count"] += 1
        session["questions"] = questions

        save_session(session)

        return {
            "answer": answer,
            "questions_used": session["question_count"],
            "questions_left": 10 - session["question_count"],
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


def guess_scientist(data: Guess):
    try:
        session = get_session(data.session_id)

        if not session:
            return {"error": "Invalid session"}, 404

        if (
            data.guess
            and data.guess.strip().lower() in session["scientist"].strip().lower()
        ):
            return {"result": True, "scientist": session["scientist"]}, 200

        return {"result": False, "scientist": session["scientist"]}, 200
    except Exception as e:
        return {"error": str(e)}, 500
