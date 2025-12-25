from fastapi import FastAPI, Depends, HTTPException, UploadFile, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.server.auth import (
    authenticate_user,
    create_jwt_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from datetime import timedelta
from fastapi.middleware.cors import CORSMiddleware

from app.langchain.chroma_store import store_document, query_documents
from typing import Union

from app.langchain.chatmodel import get_answer
from app.models.schemas import ChatRequest

from app.langchain.batuta_books import store_batuta_documents, query_batuta_documents
from app.langchain.articles import store_articles, query_articles, clean_vector_store

from app.langchain.diaralaqool import store_file, query_diaralaqool

from app.langchain.awards import store_awards_file, query_awards

from app.langchain.cinema import query_cinema

from app.langchain.scientist import (
    give_clue,
    start_game,
    ask_question,
    guess_scientist,
    AskQuestion,
    Guess,
)


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if authenticate_user(username, password):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_jwt_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/chromadb/upload")
async def create_store(
    document_id: str = Form(...),
    file: Union[str, UploadFile] = Form(None),
    text: str = Form(None),
):
    try:
        if file is None and text is None:
            raise HTTPException(
                status_code=400, detail="Either 'file' or 'text' parameter is required"
            )

        data, status_code = store_document(document_id, file, text)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "_id": data["document_id"],
                "status": data["status"],
                "status_code": status_code,
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}


@app.get("/chromadb/answer")
async def query_answer(document_id: str, query: str):
    try:
        data, status_code = query_documents(document_id, query)

        if status_code == 200:
            return {
                "question": query,
                "answer": data["answer"],
                "context": data["context"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code": 500}


@app.get("/")
async def health_check():
    return {"success": True}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        answer = get_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batuta/upload")
async def create_batuta_model(
    trip_id: str = Form(...),
    files: list[UploadFile] = Form(None),
    text: str = Form(None),
):
    try:
        if not files and text is None:
            raise HTTPException(
                status_code=400, detail="Either 'files' or 'text' parameter is required"
            )

        data, status_code = store_batuta_documents(trip_id, files, text)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "_id": data["trip_id"],
                "status": data["status"],
                "status_code": status_code,
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}


@app.get("/batuta/answer")
async def query_batuta(trip_id: str, query: str):
    try:
        data, status_code = query_batuta_documents(trip_id, query)

        if status_code == 200:
            return {
                "question": query,
                "answer": data["answer"],
                "context": data["context"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code": 500}


class ArticlesRequest(BaseModel):
    articles: list[dict]


@app.post("/articles/upload")
async def create_articles_model(request: ArticlesRequest):
    articles = request.articles
    try:
        if not articles:
            raise HTTPException(
                status_code=400, detail="The 'articles' parameter is required"
            )

        data, status_code = store_articles(articles)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "status": data["status"],
                "status_code": status_code,
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}


@app.get("/articles/answer")
async def query_articles_model(query: str):
    try:
        data, status_code = query_articles(query)

        if status_code == 200:
            return {
                "question": query,
                "answer": data["answer"],
                "answer_list": data["answer_list"],
                "context": data["context"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code": 500}


@app.post("/articles/clean")
async def clean_articles_model():
    try:
        data, status_code = clean_vector_store()

        if status_code == 200:
            return {
                "status": data["status"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        return {"error": str(e), "status_code": 500}


@app.get("/diaralaqool/answer")
async def query_diaralaqool_model(question: str):
    try:
        data, status_code = query_diaralaqool(question)

        if status_code == 200:
            return {
                "question": question,
                "answer": data["answer"],
                "context": data["context"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code": 500}


@app.post("/diaralaqool/upload")
async def create_diaralaqool_model(
    file: UploadFile = Form(...),
):
    try:
        if file is None:
            raise HTTPException(
                status_code=400, detail="The 'file' parameter is required"
            )

        data, status_code = store_file(file)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "status": data["status"],
                "status_code": status_code,
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}


@app.get("/awards/answer")
async def query_awards_model(question: str):
    try:
        data, status_code = query_awards(question)

        if status_code == 200:
            return {
                "question": question,
                "answer": data["answer"],
                "context": data["context"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code": 500}


@app.post("/awards/upload")
async def create_awards_model(
    file: UploadFile = Form(...),
):
    try:
        if file is None:
            raise HTTPException(
                status_code=400, detail="The 'file' parameter is required"
            )

        data, status_code = store_awards_file(file)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "status": data["status"],
                "status_code": status_code,
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}


@app.get("/cinema/answer")
async def query_cinema_model(query: str):
    try:
        data, status_code = query_cinema(query)
        if status_code == 200:
            return {
                "question": query,
                "answer": data["answer"],
                "status_code": status_code,
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code": 500}


@app.get("/scientist/start")
async def start_session():
    try:
        data, status_code = start_game()
        if status_code == 200:
            return {"data": data, "status_code": status_code}
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        return {"error": str(e)}, 500


@app.post("/scientist/ask")
async def ask(params: AskQuestion):
    try:
        data, status_code = ask_question(params)
        if status_code == 200:
            return {"data": data, "status_code": status_code}
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        return {"error": str(e)}, 500


@app.post("/scientist/guess")
async def guess(params: Guess):
    try:
        data, status_code = guess_scientist(params)
        if status_code == 200:
            return {"data": data, "status_code": status_code}
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/scientist/clue")
async def get_clue(session_id: str):
    try:
        data, status_code = give_clue(session_id)
        if status_code == 200:
            return {"data": data, "status_code": status_code}
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code,
            }
    except Exception as e:
        return {"error": str(e)}, 500
