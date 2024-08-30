import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
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

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Create ChromaDB client
client = chromadb.HttpClient(host="localhost", port=8000)

load_dotenv()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

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
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            documents = loader.load()
        
        else:
            # Check the file extension
            _, file_extension = os.path.splitext(file.filename)

            print(f"Received file with extension: {file_extension}")

            if file_extension.lower() not in [".docx", ".pdf", ".txt"]:
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


def store_document(document_id: str, file):
    try:
        # Extract and split text from the file
        docs = extract_text_from_file(file)
        if not docs:
            return {"error": "Failed to extract content from the file."}, 400
                
        # Create or retrieve the collection
        col = client.get_or_create_collection("documents", embedding_function=openai_ef)
        print(f"Collection created or retrieved: {col}")

        # Generate embeddings for each chunk and store them
        texts = [doc.page_content for doc in docs]
        ids = [f"{document_id}_{i}" for i in range(len(docs))]
        embeddings = openai_ef(texts)

        col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=[{"document_id": document_id}]*len(docs))
        print("Collection added successfully")

        retrieved_col=col.get(include=["metadatas"])

        return {"document_id": document_id, "status": "success"}
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}, 500
    

def query_documents(document_id: str, query: str):
    try:
        # Retrieve the collection
        col = client.get_or_create_collection("documents", embedding_function=openai_ef)

        # Generate an embedding for the query
        query_embedding = openai_ef(query)[0]

        # Retrieve documents that are similar to the query
        results = col.query(query_embeddings=[query_embedding], n_results=250, where={"document_id": document_id})

        if not results or not results["documents"]:
            return {"error": "No relevant documents found."}, 404
        
        # Prepare context from retrieved documents
        documents = [Document(page_content=doc) for doc in results["documents"][0]]

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
        answer = question_answer_chain.invoke({"context":documents,"input":query})
        return answer

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return {"error": str(e)}, 500