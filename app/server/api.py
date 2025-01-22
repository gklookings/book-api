from fastapi import FastAPI, Depends, HTTPException, UploadFile, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.langchain.excelChatmodel import get_excel_response
from app.server.auth import authenticate_user, create_jwt_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import  timedelta
from fastapi.middleware.cors import CORSMiddleware
from app.langchain.chroma_store import store_document
from app.langchain.chroma_store import query_documents
from typing import Union
from langchain.chatmodel import get_answer
from models.schemas import ChatRequest
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

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


@app.post("/vectorstore/")
async def process_directory():
    try:
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings()
        
        # Load data from the directory
        try:
            loader = UnstructuredExcelLoader("IB-AwardsList (2).xlsx", mode="elements")
            documents = loader.load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading Excel file: {str(e)}")

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Split the documents into chunks
        text_chunks = text_splitter.split_documents(documents)
        
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text chunks were created from the document")

        # Define the connection string and collection name
        CONNECTION_STRING = "postgresql+psycopg2://aibook:evaibooks_12@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books"
        COLLECTION_NAME = "excelvectorD"

        # Initialize the PGVector vector store
        vector_store = PGVector.from_documents(
            embedding=embeddings,
            documents=text_chunks,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING
        )
        
        return {"response": "VectorStore Successfully Created", "chunks_processed": len(text_chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.post("/chatexcel")
async def chatExcel(request: ChatRequest):
    try:
        answer = get_excel_response(request.question)
        return {"answerexcel": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
