from fastapi import FastAPI, Depends, HTTPException, UploadFile, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.server.auth import authenticate_user, create_jwt_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import  timedelta
from fastapi.middleware.cors import CORSMiddleware
from app.langchain.chroma_store import store_document, query_documents
from typing import Union
from app.langchain.chatmodel import get_answer
from app.models.schemas import ChatRequest
from app.langchain.batuta_books import store_batuta_documents, query_batuta_documents

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
async def create_store(document_id: str = Form(...), file: Union[str, UploadFile] = Form(None), text: str = Form(None)):
    try:
        if file is None and text is None:
            raise HTTPException(status_code=400, detail="Either 'file' or 'text' parameter is required")
        
        data, status_code = store_document(document_id, file, text)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "_id": data["document_id"],
                "status": data["status"],
                "status_code": status_code
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}
    

@app.get("/chromadb/answer")
async def query_answer(document_id:str,query:str):
    try:
        data, status_code=query_documents(document_id,query)

        if status_code == 200:
            return {
                "question": query,
                "answer": data,
                "status_code": status_code
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code":500}  

@app.get("/")
async def health_check():
    return {
        "success": True
    }
    
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        answer = get_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/batuta/upload")
async def create_batuta_model(trip_id: str = Form(...), files: list[UploadFile] = Form(None), text: str = Form(None)):
    try:
        if not files and text is None:
            raise HTTPException(status_code=400, detail="Either 'files' or 'text' parameter is required")
        
        data, status_code = store_batuta_documents(trip_id, files, text)

        # Return the success response if everything goes well
        if status_code == 200:
            return {
                "_id": data["trip_id"],
                "status": data["status"],
                "status_code": status_code
            }
        else:
            # Return the error message and code if something goes wrong
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code
            }
    except Exception as e:
        error_message = str(e)
        error_code = 500
        return {"error": error_message, "status_code": error_code}
    
@app.get("/batuta/answer")
async def query_batuta(trip_id:str,query:str):
    try:
        data, status_code=query_batuta_documents(trip_id,query)

        if status_code == 200:
            return {
                "question": query,
                "answer": data,
                "status_code": status_code
            }
        else:
            return {
                "error": data.get("error", "An error occurred"),
                "status_code": status_code
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e), "status_code":500}
