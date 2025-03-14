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
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="articles",
    connection=f"postgresql+psycopg2://{user}:{password}@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)


def store_articles(article_id: str, files, text: str):
    try:

        if text:
            # Wrap the text in a Document object
            doc = Document(page_content=text)
            all_splits = text_splitter.split_documents([doc])
        else:
            docs = extract_text_from_file(files)
            # Wrap each extracted text in a Document object
            all_splits = [Document(page_content=doc) for doc in docs]

        _ = vector_store.add_documents(documents=all_splits)

        return {"status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500


def query_articles(article_id: str, question: str):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )

        prompt_template = """
        You are a Conversational AI assistant that provides responses based on the given context.
        Respond in one of the following formats:

        1. If there is a single answer:
        [
            {{
                "answer": "ANSWER_HERE",
                "article_id": "ARTICLE_ID_HERE",
                "article_name": "ARTICLE_NAME_HERE",
                "category_id": "CATEGORY_ID_HERE",
                "category_name": "CATEGORY_NAME_HERE"
            }}
        ]

        2. If there are multiple answers:
        [
            {{
                "answer": "ANSWER_1_HERE",
                "article_id": "ARTICLE_ID_1_HERE",
                "article_name": "ARTICLE_NAME_1_HERE",
                "category_id": "CATEGORY_ID_1_HERE",
                "category_name": "CATEGORY_NAME_1_HERE"
            }},
            {{
                "answer": "ANSWER_2_HERE",
                "article_id": "ARTICLE_ID_2_HERE",
                "article_name": "ARTICLE_NAME_2_HERE",
                "category_id": "CATEGORY_ID_2_HERE",
                "category_name": "CATEGORY_NAME_2_HERE"
            }}
        ]

        3. If the answer is not present in the provided context:
        [
            {{
                "answer": "I'm sorry, but I don't have that information in the current context.",
                "article_id": "",
                "article_name": "",
                "category_id": "",
                "category_name": ""
            }}
        ]

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
        try:
            answer_json = json.loads(response["result"])
        except json.JSONDecodeError:
            answer_json = response["result"]
        return {
            "question": question,
            "answer_list": answer_json,
            "answer": {},
            "context": response["source_documents"],
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500
