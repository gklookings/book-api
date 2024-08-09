from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.langchain.get_answer import get_answer
from app.server.auth import security,authenticate_user, create_jwt_token, ACCESS_TOKEN_EXPIRE_MINUTES, verify_token, USERNAME, PASSWORD
from datetime import  timedelta
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

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

@app.get("/ask")
async def get_answers(question: str, bookId: int, token: str = Depends(security)):
    try:
        username = verify_token(token)
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        data = get_answer(question, bookId)
        return {
            "question": question,
            "answer": data['answer'],
            "context": data['context']
        }
    except Exception as e:
      # Handle the exception gracefully
      print(f"An error occurred: {e}")
      return f"An error occured. Please try again {e}"  # Or return an appropriate value indicating failure
    
@app.post("/chromadb")
async def create_store():
    try:
        my_documents = ["Hello, world!", "Hello, Chroma!"]

        # Create ChromaDB client
        client = chromadb.HttpClient(host="localhost", port=8000)
        print("ChromaDB client created successfully.")

        # Create or retrieve the collection
        col = client.get_or_create_collection("books")
        print(f"Collection created or retrieved: {col}")

        # Add documents to the collection
        col.add(ids=["1", "2"], documents=my_documents)
        print("Documents added successfully.")

        # Retrieve documents to verify they were added
        documents = col.get(ids=["1", "2"])

        # List all collections
        collections = client.list_collections()

        collection_data = [
            {"id": str(collection.id), "name": collection.name} for collection in collections
        ]
        
        return {"documents": documents, "collections":collection_data}
            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e  # Raising the exception to get the complete traceback
   

@app.get("/")
async def health_check():
    return {
        "success": True
    }