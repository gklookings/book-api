from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
# from app.llama.get_answer import get_answer
from app.langchain.get_answer import get_answer
from app.server.auth import security,authenticate_user, create_jwt_token, ACCESS_TOKEN_EXPIRE_MINUTES, verify_token, USERNAME, PASSWORD
from datetime import  timedelta
from fastapi.middleware.cors import CORSMiddleware

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
async def get_answers(question: str, token: str = Depends(security)):
    try:
        username = verify_token(token)
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        data = get_answer(question)
        # data = fetch_answer(question)
        return {
            "question": question,
            "answer": data['answer'],
            "context": data['context']
        }
    except Exception as e:
      # Handle the exception gracefully
      print(f"An error occurred: {e}")
      return "An error occured. Please try again"  # Or return an appropriate value indicating failure

@app.get("/")
async def health_check():
    return {
        "success": True
    }