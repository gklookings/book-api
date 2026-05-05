# Book API

A FastAPI-based AI-powered document intelligence and interactive game platform. The service enables document ingestion, vector-based semantic search using PostgreSQL + pgvector, conversational Q&A via OpenAI GPT-4, and an interactive scientist guessing game.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [How to Start the Project](#how-to-start-the-project)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Deployment](#deployment)

---

## Project Structure

```
book-api/
├── main.py                              # App entry point — starts Uvicorn on port 6000
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (never commit secrets)
├── .gitignore
└── app/
    ├── server/
    │   ├── api.py                       # All FastAPI routes (25 endpoints)
    │   └── auth.py                      # JWT authentication (login + token verification)
    ├── models/
    │   └── schemas.py                   # Pydantic request/response models
    └── langchain/
        ├── chroma_store.py              # Custom vector store (PostgreSQL, 768-dim embeddings)
        ├── chatmodel.py                 # RAG chain using GPT-4o + PGVector
        ├── batuta_books.py              # Travel/trip document store & retrieval
        ├── articles.py                  # Articles vector store (LangChain PGVector)
        ├── awards.py                    # Awards document store & retrieval
        ├── diaralaqool.py               # Diaralaqool book store & retrieval
        ├── cinema.py                    # Cinema Q&A with OpenAI web search
        ├── scientist.py                 # Scientist guessing game logic (two modes)
        ├── scientists_list.py           # Pool of 75 scientists for the game
        └── components/
            └── file_extractor.py        # Extracts text from DOCX, PDF, TXT, XLSX
```

---

## Architecture Overview

```
Client
  │
  ▼
FastAPI (app/server/api.py)  ←  JWT Auth (app/server/auth.py)
  │
  ├── Document Modules (langchain/)
  │     ├── Upload: file_extractor → chunk text → embed → store in PostgreSQL
  │     └── Query:  embed question → vector similarity search → LLM generates answer
  │
  ├── Scientist Game
  │     ├── Mode 1 (User guesses): Server picks scientist, user asks yes/no questions
  │     └── Mode 2 (AI guesses):  User thinks of scientist, AI asks yes/no questions
  │
  └── PostgreSQL (AWS RDS)
        ├── pgvector extension — vector similarity search
        ├── books table — generic document chunks
        ├── trips table — travel/trip document chunks
        ├── game_sessions table — guessing game state
        └── LangChain PGVector collections (articles, awards, dairalaqool)
```

**Embedding models:**

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim) — used by chroma_store and batuta_books
- `intfloat/multilingual-e5-small` — used by articles, awards, and diaralaqool modules

**LLM models:**

- `gpt-4o` / `gpt-4o-mini` — primary Q&A and game reasoning (via OpenAI)
- OpenAI Responses API with web search — used by the cinema module

---

## Features

| Feature                           | Description                                                               |
| --------------------------------- | ------------------------------------------------------------------------- |
| **Generic Document Q&A**          | Upload any file or raw text, ask questions, get AI-generated answers      |
| **Batuta (Travel) Books**         | Trip-scoped document storage and retrieval                                |
| **Articles**                      | Upload article collections (JSON), query semantically                     |
| **Awards**                        | Upload awards documents, ask questions                                    |
| **Diaralaqool**                   | Dedicated document store for Diaralaqool content                          |
| **Cinema**                        | Cinema Q&A powered by OpenAI web search                                   |
| **Scientist Game (User guesses)** | Server picks a random scientist; user asks yes/no questions, then guesses |
| **Scientist Game (AI guesses)**   | User thinks of a scientist; AI asks yes/no questions to narrow it down    |
| **JWT Authentication**            | OAuth2 password flow protecting all endpoints                             |

---

## Tech Stack

| Layer          | Technology                                |
| -------------- | ----------------------------------------- |
| Web Framework  | FastAPI 0.110, Uvicorn                    |
| AI / LLM       | OpenAI GPT-4o, LangChain                  |
| Vector Search  | PostgreSQL + pgvector, LangChain PGVector |
| Embeddings     | HuggingFace sentence-transformers         |
| Database ORM   | SQLAlchemy 2.0                            |
| Authentication | python-jose (JWT), OAuth2 password flow   |
| File Parsing   | pypdf, docx2txt, pandas (XLSX)            |
| Runtime        | Python 3.9+                               |

---

## Prerequisites

- Python 3.9 or higher
- PostgreSQL with the `pgvector` extension enabled (or use the configured AWS RDS instance)
- An OpenAI API key

---

## Environment Setup

Create a `.env` file in the project root with the following variables:

```env
# PostgreSQL connection string (with pgvector extension)
CONNECTION_STRING=postgresql+psycopg2://<user>:<password>@<host>/<dbname>

# OpenAI
OPENAI_API_KEY=sk-...

# JWT Auth
SECRET_KEY=your-secret-key
USER_NAME=your-username
PASSWORD=your-password

# Vector store collection name
COLLECTION_NAME=poems_vector

# Optional: Groq (defined in code but not currently active)
GROQ_API_KEY=...
LLM_MODEL_NAME=llama-3.1-70b-versatile
```

> **Note:** Never commit real API keys or database credentials to version control.

---

## How to Start the Project

### 1. Clone the repository

```bash
git clone <repo-url>
cd book-api
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Some packages (e.g. `unstructured[pdf]`) may require system-level dependencies like `libmagic`. On macOS: `brew install libmagic`.

### 4. Configure environment variables

Copy the template above into a `.env` file and fill in real values.

### 5. Start the server

```bash
python main.py
```

The server starts on **http://0.0.0.0:6000** with hot-reload enabled.

### Interactive API docs

| Interface  | URL                         |
| ---------- | --------------------------- |
| Swagger UI | http://localhost:6000/docs  |
| ReDoc      | http://localhost:6000/redoc |

---

## API Endpoints

All endpoints are open and do not require authentication.

---

### Health Check

| Method | Path | Description                |
| ------ | ---- | -------------------------- |
| `GET`  | `/`  | Returns `{"status": "ok"}` |

---

### Generic Document Store (ChromaDB / PostgreSQL)

| Method | Path               | Description                                            |
| ------ | ------------------ | ------------------------------------------------------ |
| `POST` | `/chromadb/upload` | Upload a file or raw text; stores chunks + embeddings  |
| `GET`  | `/chromadb/answer` | Query stored documents by `document_id` and `question` |

---

### Batuta (Travel) Books

| Method | Path             | Description                                        |
| ------ | ---------------- | -------------------------------------------------- |
| `POST` | `/batuta/upload` | Upload multiple files scoped to a `trip_id`        |
| `GET`  | `/batuta/answer` | Query trip documents with `trip_id` and `question` |

---

### Articles

| Method | Path               | Description                                                                         |
| ------ | ------------------ | ----------------------------------------------------------------------------------- |
| `POST` | `/articles/upload` | Upload articles as a JSON array `[{article_id, article_name, article_description}]` |
| `GET`  | `/articles/answer` | Query articles by `question`                                                        |
| `POST` | `/articles/clean`  | Clear the entire articles vector store                                              |

---

### Diaralaqool

| Method | Path                  | Description                   |
| ------ | --------------------- | ----------------------------- |
| `POST` | `/diaralaqool/upload` | Upload a single file          |
| `GET`  | `/diaralaqool/answer` | Query documents by `question` |

---

### Awards

| Method | Path             | Description                     |
| ------ | ---------------- | ------------------------------- |
| `POST` | `/awards/upload` | Upload an awards file           |
| `GET`  | `/awards/answer` | Query awards data by `question` |

---

### Cinema

| Method | Path             | Description                                      |
| ------ | ---------------- | ------------------------------------------------ |
| `GET`  | `/cinema/answer` | Query cinema information using OpenAI web search |

---

### Scientist Guessing Game

#### Mode 1 — User Guesses the Scientist

The server secretly picks a scientist. The user asks yes/no questions to narrow it down, then makes a guess.

| Method | Path               | Description                                           |
| ------ | ------------------ | ----------------------------------------------------- |
| `GET`  | `/scientist/start` | Start a new game session, returns a `session_id`      |
| `POST` | `/scientist/ask`   | Ask a yes/no question; body: `{session_id, question}` |
| `POST` | `/scientist/guess` | Guess the scientist; body: `{session_id, guess}`      |
| `GET`  | `/scientist/clue`  | Get a clue; params: `session_id`                      |

**Flow:**

```
GET /scientist/start
  → { session_id: "abc123" }

POST /scientist/ask  { session_id: "abc123", question: "Was this scientist a physicist?" }
  → { answer: "Yes", questions_remaining: 19 }

GET /scientist/clue?session_id=abc123
  → { clue: "This scientist..." }

POST /scientist/guess  { session_id: "abc123", guess: "Albert Einstein" }
  → { correct: true, scientist: "Albert Einstein" }
```

#### Mode 2 — AI Guesses the Scientist

The user thinks of a scientist. The AI asks yes/no questions and eventually makes a guess.

| Method | Path                         | Description                                                                                              |
| ------ | ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| `GET`  | `/scientist/ai-guess/start`  | Start AI-guessing session, returns a `session_id` and the AI's first question                            |
| `POST` | `/scientist/ai-guess/answer` | Answer the AI's question; body: `{session_id, answer: "yes"/"no"}` — returns next question or AI's guess |

**Flow:**

```
GET /scientist/ai-guess/start
  → { session_id: "xyz456", question: "Is your scientist alive today?" }

POST /scientist/ai-guess/answer  { session_id: "xyz456", answer: "no" }
  → { question: "Was your scientist born before 1900?", status: "ongoing" }

... (repeat)

POST /scientist/ai-guess/answer  { session_id: "xyz456", answer: "yes" }
  → { guess: "Marie Curie", status: "guessing" }
```

---

## Database Schema

The application uses PostgreSQL with pgvector. Tables are created automatically on first use.

| Table                 | Columns                                                                              | Purpose                                       |
| --------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------- |
| `books`               | `bookid`, `text_content`, `embedding_vector (768)`                                   | Generic document chunks                       |
| `trips`               | `tripid`, `text_content`, `embedding_vector (768)`                                   | Travel/trip document chunks                   |
| `game_sessions`       | `session_id (PK)`, `scientist`, `question_count`, `questions (JSON)`, `clues (JSON)` | Guessing game state                           |
| LangChain collections | managed by LangChain                                                                 | Used by articles, awards, diaralaqool modules |

---

## Running Locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your .env file (see Environment Setup above)

# 4. Start the server
python main.py
```

Server runs at **http://localhost:6000**. Interactive docs at **http://localhost:6000/docs**.

---

## Deployment

The production server runs on **AWS Lightsail** using a **Bitnami Apache** stack with **PM2** as the process manager.

### Stack

| Component      | Role                                                                   |
| -------------- | ---------------------------------------------------------------------- |
| AWS Lightsail  | Cloud VM hosting the application                                       |
| Bitnami Apache | Web server / reverse proxy (forwards HTTP → Uvicorn on port 6000)      |
| PM2            | Node-based process manager that keeps the Python/Uvicorn process alive |

### PM2 Commands (on the server)

```bash
# View running processes
pm2 list

# Start the app - only needed when starting as new
pm2 start main.py --interpreter python3 --name aibook

# Restart after a code change
pm2 restart aibook

# View live logs
pm2 logs aibook

# Stop the app
pm2 stop aibook

# Save process list so it survives reboots
pm2 save
pm2 startup
```

### Deploying code changes

```bash
# On the Lightsail instance
cd ev-ai/book-api
git pull
pip install -r requirements.txt   # only if dependencies changed
pm2 restart aibook
```

### Apache reverse proxy

Apache is configured to forward incoming requests to Uvicorn running locally on port 6000. The Bitnami Apache config directory is typically at `/opt/bitnami/apache/conf/`.
