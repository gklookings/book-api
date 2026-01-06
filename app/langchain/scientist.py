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


def start_ai_guessing_game():
    """Start a new AI guessing game where AI tries to guess the scientist the user is thinking of."""
    session_id = str(uuid.uuid4())
    try:
        session = {
            "session_id": session_id,
            "scientist": "AI_GUESSING_MODE",  # Placeholder to satisfy NOT NULL constraint
            "question_count": 0,
            "questions": [],
            "clues": [],
        }

        save_session(session)

        # Generate and return the first question immediately
        scientists_list = ", ".join(SCIENTISTS)

        messages = [
            {
                "role": "system",
                "content": """You are playing a guessing game. The user is thinking of a scientist from this list:
                
"""
                + scientists_list
                + """

You must ask strategic yes/no questions to figure out which scientist they are thinking of.
Ask questions that will help narrow down the possibilities efficiently.
Only ask one question at a time.
Make your questions clear and answerable with yes or no.""",
            },
            {
                "role": "user",
                "content": f"""This is the first question. Please ask a yes/no question to figure out which scientist the user is thinking of.""",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4.1", messages=messages, temperature=0.7
        )

        first_question = response.choices[0].message.content.strip()

        # Save the first question to session
        session["questions"] = [{"q": first_question, "a": None}]
        session["question_count"] = 1
        save_session(session)

        return {
            "message": "AI guessing game started. Think of a scientist!",
            "session_id": session_id,
            "question": first_question,
            "question_number": 1,
            "questions_left": 19,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


class AIGuessQuestion(BaseModel):
    session_id: str
    answer: str  # "yes" or "no"


def answer_ai_question(data: AIGuessQuestion):
    """Record the user's answer to the AI's question and generate next question or make final guess."""
    try:
        session = get_session(data.session_id)

        if not session:
            return {"error": "Invalid session"}, 404

        if session["question_count"] > 20:
            return {"error": "No more questions allowed"}, 400

        # Validate answer
        answer = data.answer.strip().lower()
        if answer not in ["yes", "no"]:
            return {"error": "Answer must be 'yes' or 'no'"}, 400

        # Store the answer to the last question
        questions = session["questions"] or []
        if questions:
            questions[-1]["a"] = answer

        session["questions"] = questions

        save_session(session)

        # If 20 questions reached, make a guess
        if session["question_count"] >= 20:
            # Build context from all questions and answers
            questions_context = ""
            for qa in session["questions"]:
                questions_context += f"Q: {qa['q']}\nA: {qa['a']}\n"

            scientists_list = ", ".join(SCIENTISTS)

            messages = [
                {
                    "role": "system",
                    "content": """You are playing a guessing game. Based on the Q&A history, you need to guess which scientist 
the user is thinking of. Choose from the provided list of scientists. 
Return ONLY the name of the scientist, nothing else.""",
                },
                {
                    "role": "user",
                    "content": f"""Here is the conversation history:

{questions_context}

Available scientists to choose from:
{scientists_list}

Which scientist am I thinking of? Answer with just the name.""",
                },
            ]

            response = client.chat.completions.create(
                model="gpt-4.1", messages=messages, temperature=0
            )

            guess = response.choices[0].message.content.strip()

            return {
                "message": "20 questions completed! Here's my final guess:",
                "guess": guess,
                "questions_used": session["question_count"],
                "game_finished": True,
            }, 200

        # Generate next question
        questions_context = "Previous Q&A:\n"
        for qa in session["questions"]:
            questions_context += f"Q: {qa['q']}\nA: {qa['a']}\n"

        scientists_list = ", ".join(SCIENTISTS)

        messages = [
            {
                "role": "system",
                "content": """You are playing a guessing game. The user is thinking of a scientist from this list:
                
"""
                + scientists_list
                + """

You must ask strategic yes/no questions to figure out which scientist they are thinking of.
Ask questions that will help narrow down the possibilities efficiently.
Only ask one question at a time.
Make your questions clear and answerable with yes or no.

You can also make a guess if you are confident. If the user answered "yes" to your previous guess, return game_finished as true.
If the user answered "yes" to your previous guess, return question_or_guess as thanking the user for playing and your final guess.

Return your response as JSON with this format:
{"question_or_guess": "your question or guess here", "game_finished": false}

If the user answered yes to your guess, set game_finished to true.""",
            },
            {
                "role": "user",
                "content": f"""Question number: {session['question_count'] + 1} out of 20.
The user answered "{answer}" to the previous question.
                
{questions_context}

Please ask the next yes/no question or make a guess if confident. Return as JSON.""",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4.1", messages=messages, temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()
        print(f"AI Response: {response_text}")

        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            next_question = response_json.get("question_or_guess", response_text)
            game_finished = response_json.get("game_finished", False)
        except json.JSONDecodeError:
            next_question = response_text
            game_finished = False

        # Save the next question to session
        session["questions"].append({"q": next_question, "a": None})
        session["question_count"] += 1
        save_session(session)

        return {
            "question": next_question,
            "question_number": session["question_count"],
            "questions_used": session["question_count"],
            "questions_left": 20 - session["question_count"],
            "game_finished": game_finished,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500
