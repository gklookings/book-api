from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.llama.get_answer import get_answer
from app.server.auth import security,authenticate_user, create_jwt_token, ACCESS_TOKEN_EXPIRE_MINUTES, verify_token, USERNAME, PASSWORD
from datetime import  timedelta


app = FastAPI()

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
@app.get("/")
async def get_answers(question: str, token: str = Depends(security)):
    print(token)
    username = verify_token(token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    response = get_answer(question)
    return {
        "question": question,
        "answer": response
    }