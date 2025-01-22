import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from psycopg2 import extras
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import json
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from app.langchain.chroma_store import split_text_into_chunks, extract_text_from_file

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
sentence_transformer_ef = SentenceTransformer(embedding_model)

def store_articles(article_id: str, files, text: str):
    try:
        texts = []

        if text:
            texts = split_text_into_chunks(text)
            if not texts or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                return {"error": "Texts must be a list of valid strings."}, 400
        if files:
            for file in files:
                docs = extract_text_from_file(file)
                if not docs:
                    return {"error": f"Failed to extract content from the file: {file}"}, 400
                    
                for doc in docs:
                    text_chunks = split_text_into_chunks(doc.page_content)
                    texts.extend(text_chunks)
            
            if not texts or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                return {"error": "Texts must be a list of valid strings."}, 400

        print(f"Texts extracted and split into chunks: {len(texts)}")

        print(f"Generating embeddings for {len(texts)} chunks")

        def batch_list(lst, batch_size):
            for i in range(0, len(lst), batch_size):
                yield lst[i:i + batch_size]

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
            connect_timeout=600
        )
        cur = conn.cursor()

        print("Connected to the database")

        # Insert new data
        query = """
            INSERT INTO articles (article_id, text_content, embedding_vector)
            VALUES %s
        """
        values = [
            (article_id, texts[i], np.array(embeddings[i], dtype=np.float32).tolist()) for i in range(len(texts))
        ]

        try:
            extras.execute_values(cur, query, values, page_size=batch_size)
            conn.commit()
            print("Data inserted successfully")
        finally:
            cur.close()
            conn.close()

        return {"article_id": article_id, "status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500
    
def query_articles(article_id: str, query: str):
    try:
        # Acquire a connection from the pool
        conn = psycopg2.connect(
            host="ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com",
            database="books",
            user=user,
            password=password,
            connect_timeout=600
        )
        if conn is None:
            return {"error": "Unable to acquire a connection from the pool."}, 500
        cur = conn.cursor()

        print("Connected to the database")

        # Generate embedding for the query
        query_embedding = sentence_transformer_ef.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Perform vector similarity search using pgvector
        sql_query = """
        SELECT embedding_vector, text_content
        FROM articles
        WHERE article_id = %s
        ORDER BY embedding_vector <=> %s::vector
        LIMIT 230;
        """
        cur.execute(sql_query, (article_id, query_embedding.tolist()))
        rows = cur.fetchall()

        # Handle no results
        if not rows:
            return {"error": "No relevant documents found."}, 404

        # Parse results
        doc_embeddings = [np.array(ast.literal_eval(row[0]), dtype=np.float32) for row in rows]
        document_texts = [row[1] for row in rows]

        # Calculate similarity scores and format top results
        top_documents = []

        # Set a threshold for "relevance"
        for i in range(len(doc_embeddings)):
            similarity = 1 - np.dot(query_embedding, doc_embeddings[i]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embeddings[i])
            )
            if similarity > 0.5:  # Adjust this threshold as needed
                # Only include documents that pass this threshold
                top_documents.append({
                    "page_content": document_texts[i],
                    "metadata": {"similarity": similarity}
                })

        print(f"Retrieved {len(top_documents)} relevant documents")

        # Prepare documents for LangChain
        documents = [Document(page_content=doc["page_content"]) for doc in top_documents]

        # Create LangChain QA chain
        system_prompt = """
        You are an intelligent assistant. Use the given context to answer the question as accurately as possible. 
        If the answer is not explicitly in the context, make an educated guess.

        Format the answer based on the query:
        - If the question requires a single response, return the answer in this JSON format:
        {{
            "answer": "string",
            "article_id": "string",
            "article_name": "string",
            "category_id": "string",
            "category_name": "string"
        }}
        - If the question requires a list of objects, return the answer as a JSON array:
        [
            {{
                "answer": "string",
                "article_id": "string",
                "article_name": "string",
                "category_id": "string",
                "category_name": "string"
            }},
            {{
                "answer": "string",
                "article_id": "string",
                "article_name": "string",
                "category_id": "string",
                "category_name": "string"
            }}
            // Add more objects if needed
        ]

        If you cannot answer the question based on the context, respond with:
        - For a single response:
        {{
            "answer": "I don't know",
            "article_id": "",
            "article_name": "",
            "category_id": "",
            "category_name": ""
        }}
        - For a list-based query:
        []

        Context: {context}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # Generate answer
        answer = question_answer_chain.invoke({"context": documents, "input": query})

        print(f"Answer generated: {answer}")

        # Parse the answer to JSON
        try:
            # Clean the answer by removing "```json" formatting if present
            cleaned_answer = answer.strip("```json").strip() if "```json" in answer else answer.strip()
            
            # Parse the JSON string into a Python object (could be a dict or list)
            parsed_answer = json.loads(cleaned_answer)
            
            # Check if the parsed answer is a list
            if isinstance(parsed_answer, list):
                # Construct the response for a list of answers
                response_list = [
                    {
                        "answer": item.get("answer", "I don't know"),
                        "article_id": item.get("article_id", ""),
                        "article_name": item.get("article_name", ""),
                        "category_id": item.get("category_id", ""),
                        "category_name": item.get("category_name", "")
                    }
                    for item in parsed_answer
                ]
                response={}
            elif isinstance(parsed_answer, dict):
                # Construct the response for a single answer
                response = {
                    "answer": parsed_answer.get("answer", "I don't know"),
                    "article_id": parsed_answer.get("article_id", ""),
                    "article_name": parsed_answer.get("article_name", ""),
                    "category_id": parsed_answer.get("category_id", ""),
                    "category_name": parsed_answer.get("category_name", "")
                }
                response_list=[]
            else:
                raise ValueError("Invalid JSON format: Expected a list or object.")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in the answer: {answer}. Error: {e}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while processing the answer: {e}")

        return {"answer": response, "answer_list": response_list, "context": top_documents[:5]}, 200

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return {"error": str(e)}, 500

    finally:
        # Ensure resources are cleaned up
        if 'cur' in locals() and not cur.closed:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()