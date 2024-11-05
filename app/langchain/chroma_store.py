import os
import chromadb.utils.embedding_functions as embedding_functions
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import tempfile
import os
from dotenv import load_dotenv
from typing import Union
import requests
from transformers import AutoTokenizer
import psycopg2
import numpy as np

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

# Function to split text into chunks using Hugging Face tokenizer
def split_text_into_chunks(text, model_name=embedding_model, max_tokens=512):
    # Load the appropriate tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)  # Disable adding special tokens
    
    # Split tokens into chunks that fit within the max token limit of 512 tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))  # Decode tokens back into text for each chunk
    
    return chunks

chunk_size = 1000
chunk_overlap = 100

def extract_text_from_file(file: Union[UploadFile, str]):
    try:
        if isinstance(file, str):
            # Download the file from the URL
            response = requests.get(file)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download file from URL")

            # Create a temporary file to store the downloaded content
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            print(f"Extracting text from the file located at: {temp_file_path}")

            # Load the content based on the file extension
            _, file_extension = os.path.splitext(file)
            if file_extension.lower() == ".docx":
                loader = Docx2txtLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            documents = loader.load()
        
        else:
            # Check the file extension
            _, file_extension = os.path.splitext(file.filename)

            print(f"Received file with extension: {file_extension}")

            if file_extension.lower() not in [".docx", ".pdf", ".txt", ".xlsx"]:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file.file.read())
                temp_file_path = temp_file.name

            print(f"Extracting text from the file located at: {temp_file_path}")

            # Load the content based on the file type
            if file_extension.lower() == ".docx":
                loader = Docx2txtLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            documents = loader.load()

        # Split the document into chunks
        doc_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = doc_splitter.split_documents(documents)

        return docs

    except HTTPException as e:
        raise e  # Propagate the HTTPException
    except Exception as e:
        print(f"Failed to extract text from the file: {e}")
        return None


def store_document(document_id: str, file, text: str):
    try:
        texts = []

        if text:
            # Split the provided text into chunks
            texts = split_text_into_chunks(text)
            if not texts or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                return {"error": "Texts must be a list of valid strings."}, 400
        else:
            # Extract and split text from the uploaded file
            docs = extract_text_from_file(file)
            if not docs:
                return {"error": "Failed to extract content from the file."}, 400
                    
            # Chunk the extracted text based on token limits
            for doc in docs:
                text_chunks = split_text_into_chunks(doc.page_content)
                texts.extend(text_chunks)  # Add all chunks to the list
            
            if not texts or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                return {"error": "Texts must be a list of valid strings."}, 400

        print(f"Texts extracted and split into chunks: {len(texts)}")

        print(f"Generating embeddings for {len(texts)} chunks")

        # Helper function to batch large lists
        def batch_list(lst, batch_size):
            for i in range(0, len(lst), batch_size):
                yield lst[i:i + batch_size]

        batch_size = 1000  # Adjust batch size as needed
        embeddings = []
        for batch in batch_list(texts, batch_size):
            batch_embeddings = sentence_transformer_ef(batch)
            embeddings.extend(batch_embeddings)

        if not embeddings or len(embeddings) != len(texts):
            return {"error": "Failed to generate embeddings."}, 500

        print(f"Embeddings generated")

        # Prepare the SQL query for inserting data
        query = """
            INSERT INTO books (bookId, text_content, embedding)
            VALUES (%s, %s, %s)
        """

        # Establish PostgreSQL connection
        conn = psycopg2.connect(
            host="ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com",
            database="books",
            user=user,
            password=password,
            connect_timeout=600
        )
        cur = conn.cursor()

        # Clear previous records related to the document_id if needed
        cur.execute("DELETE FROM books WHERE bookId = %s", (document_id,))
        conn.commit()
        print("Previous records deleted")

        # Insert each document chunk and its corresponding embedding into the database
        for i in range(len(texts)):
            cur.execute(
                query,
                (document_id, texts[i] ,np.array(embeddings[i], dtype=float).tolist())
            )
            print(f"Inserted chunk {i + 1}")

        # Commit the transaction
        conn.commit()
        print("Data inserted successfully")

        # Close the cursor and connection
        cur.close()
        conn.close()

        return {"document_id": document_id, "status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500
    

def query_documents(document_id: str, query: str):
    try:
        # Establish PostgreSQL connection
        conn = psycopg2.connect(
            host="ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com",
            database="books",
            user=user,
            password=password,
            connect_timeout=600
        )
        cur = conn.cursor()

        # Retrieve the relevant documents from the database
        cur.execute("SELECT * FROM books WHERE bookId = %s;", (document_id,))
        rows = cur.fetchall()

        if not rows:
            return {"error": "No relevant documents found."}, 404
        
        # Extract the document content and embeddings from the fetched rows
        documents = [row[2] for row in rows]  # Assuming column 2 is embedding
        document_texts = [row[3] for row in rows]  # Assuming column 3 is text content

        # Generate embedding for the query
        query_embedding = sentence_transformer_ef(query)[0]

        # Generate embeddings for the documents and match them with the query embedding
        doc_embeddings = [np.array(doc, dtype=float) for doc in documents]  # Convert embeddings to numpy arrays

        # Find the most similar documents based on embeddings
        similarities = [np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                        for doc_embedding in doc_embeddings]

        # Get the indices of the top most similar documents
        search_value = -50 if document_id == 'IB-AwardsList' else -30
        top_indices = np.argsort(similarities)[search_value:][::-1]

        # Prepare context from the top documents
        top_documents = [Document(page_content=document_texts[i]) for i in top_indices]
        print(f"Retrieved {len(top_documents)} relevant documents")

        # Create a retrieval-based QA chain using LangChain
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # Generate the answer to the query
        answer = question_answer_chain.invoke({"context": top_documents, "input": query})

        # Close the cursor and connection
        cur.close()
        conn.close()

        return answer, 200

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return {"error": str(e)}, 500