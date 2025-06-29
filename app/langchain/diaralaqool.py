import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from app.langchain.components.file_extractor import extract_text_from_file
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.docstore.document import Document
from fastapi import UploadFile


os.environ["TOKENIZERS_PARALLELISM"] = "false"
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

llm = init_chat_model("gpt-4o", model_provider="openai")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="dairalaqool",
    connection=f"postgresql+psycopg2://{user}:{password}@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)


def clean_vector_store():
    try:
        # Clean the vector store by removing all documents
        vector_store.delete_collection()
        vector_store.create_collection()
        print("Vector store cleaned and collection recreated.")
        return {"status": "success"}, 200
    except Exception as e:
        return {"error": str(e)}, 500


def store_file(file: UploadFile):
    try:
        documents = []

        # Extract text from the uploaded file
        extracted_text = extract_text_from_file(file)
        if not extracted_text:
            return {"error": "No text extracted from the file."}, 400

        # Split the text into chunks
        for doc in extracted_text:
            chunks = text_splitter.split_text(doc.page_content)

        # Create Document objects for each chunk
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
            )
            documents.append(doc)

        # Store documents with metadata
        print(f"Storing {len(documents)} documents in the vector store.")
        _ = vector_store.add_documents(documents=documents)

        return {"status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500


def query_diaralaqool(question: str):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 200}
        )

        prompt_template = """
        You are a Conversational AI assistant that provides responses based on the given context.
        Only use the information from the context to answer the question.
        Your answer should be concise and relevant to the question asked.
        If the question is not related to the context, respond with "I don't know".

        Context:
        {context}

        Question:
        {question}
        """

        chat_prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={"prompt": chat_prompt},
        )

        response = qa_chain.invoke(question)

        return {
            "question": question,
            "answer": response["result"],
            "context": response["source_documents"],
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500
