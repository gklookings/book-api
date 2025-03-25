import os
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
from typing import Union
import requests
from transformers import AutoTokenizer
from psycopg2 import extras
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.chat_models import init_chat_model


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
sentence_transformer_ef = SentenceTransformer(embedding_model)


# Function to split text into chunks using Hugging Face tokenizer
def split_text_into_chunks(
    text, model_name=embedding_model, max_tokens=512, overlap=50
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Tokenized text into {len(tokens)} tokens")
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i : i + max_tokens]
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(decoded_chunk)
        print(f"Decoded chunk: {decoded_chunk}, chunk length: {len(chunk)}")
        if i + max_tokens >= len(tokens):
            break
    return chunks


chunk_size = 1000
chunk_overlap = 100


def extract_text_from_file(file: Union[UploadFile, str]):
    try:
        if isinstance(file, str):
            response = requests.get(file)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail="Failed to download file from URL"
                )
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            print(f"Extracting text from the file located at: {temp_file_path}")
            _, file_extension = os.path.splitext(file)
            loader = None
            if file_extension.lower() == ".docx":
                loader = Docx2txtLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
            if loader:
                documents = loader.load()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

        else:
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".docx", ".pdf", ".txt", ".xlsx"]:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(file.file.read())
                temp_file_path = temp_file.name

            print(f"Extracting text from the file located at: {temp_file_path}")
            loader = None
            if file_extension.lower() == ".docx":
                loader = Docx2txtLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
            if loader:
                documents = loader.load()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

        print(f"Text extracted from the file: {len(documents)} documents")

        doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = doc_splitter.split_documents(documents)

        return docs

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Failed to extract text from the file: {e}")
        return None


def store_document(document_id: str, file, text: str):
    try:
        texts = []

        if text:
            texts = split_text_into_chunks(text)
            if (
                not texts
                or not isinstance(texts, list)
                or not all(isinstance(text, str) for text in texts)
            ):
                return {"error": "Texts must be a list of valid strings."}, 400
        if file:
            docs = extract_text_from_file(file)
            if not docs:
                return {"error": "Failed to extract content from the file."}, 400

            for doc in docs:
                text_chunks = split_text_into_chunks(doc.page_content)
                texts.extend(text_chunks)

            if (
                not texts
                or not isinstance(texts, list)
                or not all(isinstance(text, str) for text in texts)
            ):
                return {"error": "Texts must be a list of valid strings."}, 400

        print(f"Texts extracted and split into chunks: {len(texts)}")

        print(f"Generating embeddings for {len(texts)} chunks")

        def batch_list(lst, batch_size):
            for i in range(0, len(lst), batch_size):
                yield lst[i : i + batch_size]

        batch_size = 1000
        embeddings = []

        for batch in batch_list(texts, batch_size):
            batch_embeddings = sentence_transformer_ef.encode(batch)
            embeddings.extend(batch_embeddings)

        if not embeddings or len(embeddings) != len(texts):
            return {"error": "Failed to generate embeddings."}, 500

        print(f"Embeddings generated")

        embedding_dim = 768  # Ensure embeddings are 768-dimensional
        if not all(len(e) == embedding_dim for e in embeddings):
            return {"error": "Embeddings are not of fixed dimensions."}, 500

        conn = psycopg2.connect(
            host="ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com",
            database="books",
            user=user,
            password=password,
            connect_timeout=600,
        )
        cur = conn.cursor()

        print("Connected to the database")

        # If data exists for this bookId, delete it
        cur.execute("DELETE FROM books WHERE bookid = %s", (document_id,))
        conn.commit()
        print("Old data removed if it existed.")

        # Insert new data
        query = """
            INSERT INTO books (bookid, text_content, embedding_vector)
            VALUES %s
        """
        values = [
            (document_id, texts[i], np.array(embeddings[i], dtype=np.float32).tolist())
            for i in range(len(texts))
        ]

        try:
            extras.execute_values(cur, query, values, page_size=batch_size)
            conn.commit()
            print("Data inserted successfully")
        finally:
            cur.close()
            conn.close()

        return {"document_id": document_id, "status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500


def query_documents(document_id: str, query: str):
    try:
        # Acquire a connection from the pool
        conn = psycopg2.connect(
            host="ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com",
            database="books",
            user=user,
            password=password,
            connect_timeout=600,
        )
        if conn is None:
            return {"error": "Unable to acquire a connection from the pool."}, 500
        cur = conn.cursor()

        # Generate embedding for the query
        query_embedding = sentence_transformer_ef.encode(
            query
        )  # Convert the query to an embedding

        # Convert query_embedding to a numpy array of type np.float32
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Perform vector similarity search using pgvector
        if document_id == "IB-AwardsList":
            limit = 500
        else:
            limit = 150

        sql_query = f"""
        SELECT embedding_vector, text_content
        FROM books
        WHERE bookid = %s
        ORDER BY embedding_vector <=> %s::vector  -- Explicitly cast to vector
        LIMIT {limit};
        """

        # Execute the query with adapted query embedding (passing it as a numpy array directly)
        cur.execute(sql_query, (document_id, query_embedding.tolist()))
        rows = cur.fetchall()

        # Ensure there are results
        if not rows:
            return {"error": "No relevant documents found."}, 404

        # Extract embeddings and text content from the fetched rows
        doc_embeddings = [
            np.array(ast.literal_eval(row[0]), dtype=np.float32) for row in rows
        ]
        document_texts = [row[1] for row in rows]

        # Sort by similarity and get the top results
        top_documents = [
            {
                "page_content": document_texts[i],
                "metadata": {
                    "similarity": 1
                    - np.dot(query_embedding, doc_embeddings[i])
                    / (
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(doc_embeddings[i])
                    )
                },
            }
            for i in range(len(doc_embeddings))
        ]
        print(f"Retrieved {len(top_documents)} relevant documents")

        # Extract only the page content for LangChain
        document_contents = [doc["page_content"] for doc in top_documents]
        documents = [
            Document(page_content=document_texts[i])
            for i in range(len(document_contents))
        ]

        # Create a retrieval-based QA chain using LangChain
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # Generate the answer to the query
        answer = question_answer_chain.invoke({"context": documents, "input": query})

        # Close the cursor and release the connection back to the pool
        cur.close()
        conn.close()

        return {"answer": answer, "context": top_documents[:5]}, 200

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return {"error": str(e)}, 500
