import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from app.langchain.components.file_extractor import extract_text_from_file
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from fastapi import UploadFile
import re


os.environ["TOKENIZERS_PARALLELISM"] = "false"
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

llm = init_chat_model("gpt-4o", model_provider="openai")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="motanabi",
    connection=f"postgresql+psycopg2://{user}:{password}@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)


def clean_vector_store():
    try:
        # Clean the vector store by removing all documents in the motanabi collection only
        vector_store.delete_collection()
        vector_store.create_collection()
        print("Motanabi vector store cleaned and collection recreated.")
        return {"status": "success"}, 200
    except Exception as e:
        return {"error": str(e)}, 500


def store_file(file: UploadFile):
    try:
        # Extract text from the uploaded file
        extracted_text = extract_text_from_file(file)
        if not extracted_text:
            return {"error": "No text extracted from the file."}, 400

        # Split all documents/pages into chunks at once, preserving metadata
        documents = text_splitter.split_documents(extracted_text)

        # Store documents with metadata
        print(f"Storing {len(documents)} documents in the motanabi vector store.")
        _ = vector_store.add_documents(documents=documents)

        return {"status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500


def query_motanabi(question: str):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 30}
        )

        prompt_template = """
        You are a Conversational AI assistant that provides responses based on the given context.
        Only use the information from the context to answer the question.
        If the question is not related to the context, respond with "I don't know".

        Critical Instructions (follow exactly):
        - Return poem lines that are semantically or thematically related to the subject asked about in the question.
          This includes lines that directly mention the subject AND lines whose meaning or imagery is clearly about the topic.
        - Do NOT mix information from different poems.
        - You MUST reference the poemId for every line you mention using this EXACT format: [poemId: ID]
          Example: "وَقَفتُ عَلى رَبعٍ لِمَيَّةَ" [poemId: 42]
          NEVER use any other format such as "PoemId: 42" or "(poemId: 42)" — brackets are mandatory.
        - If no lines in the context are related to the subject, say "I don't know".

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
        answer = response["result"]

        # Extract poemIds using regex (case-insensitive to handle LLM format variations)
        poem_ids = re.findall(r"\[poemId:\s*(\w+)\]", answer, re.IGNORECASE)
        # Remove poemId markers from the answer to clean it
        cleaned_answer = re.sub(r"\[poemId:\s*\w+\]", "", answer, flags=re.IGNORECASE).strip()

        return {
            "question": question,
            "answer": cleaned_answer,
            "poemIds": list(set(poem_ids)),  # Return unique poemIds
            "context": response["source_documents"],
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500
