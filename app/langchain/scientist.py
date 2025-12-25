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
                INSERT INTO game_sessions (session_id, scientist, question_count, questions, clues)
                VALUES (:session_id, :scientist, :question_count, :questions, :clues)
                ON CONFLICT (session_id)
                DO UPDATE SET
                    scientist = EXCLUDED.scientist,
                    question_count = EXCLUDED.question_count,
                    questions = EXCLUDED.questions,
                    clues = EXCLUDED.clues
            """
            ),
            {
                "session_id": session["session_id"],
                "scientist": session["scientist"],
                "question_count": session["question_count"],
                "questions": json.dumps(session["questions"]),
                "clues": session["clues"],
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
            "clues": [],
        }

        save_session(session)

        return {
            "message": "Game started",
            "questions_left": 20,
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

        if session["question_count"] >= 20:
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
            "questions_left": 20 - session["question_count"],
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


def give_clue(session_id: str):
    try:
        session = get_session(session_id)

        if not session:
            return {"error": "Invalid session"}, 404

        scientist = session["scientist"]
        clues = session["clues"] or []

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Provide a brief clue about the scientist: {scientist}.  
                            Keep it concise. 
                            Do not reveal the name of the scientist. 
                            Do not make up any facts. 
                            The clues already given are: {', '.join(clues) if clues else 'None'}. Do not repeat any of these.
                            Do not make it too obvious.
                            """,
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4.1", messages=messages, temperature=0.7
        )

        clue = response.choices[0].message.content.strip()
        clues.append(clue)
        session["clues"] = clues

        save_session(session)

        return {"clue": clue}, 200
    except Exception as e:
        return {"error": str(e)}, 500
