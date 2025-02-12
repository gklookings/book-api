import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from psycopg2 import extras
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from app.langchain.chroma_store import split_text_into_chunks, extract_text_from_file
from psycopg2 import pool
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import concurrent.futures
from scipy.spatial.distance import cdist

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

# Initialize a connection pool
db_pool = pool.SimpleConnectionPool(
    minconn=1, maxconn=100,
    host="ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com",
    database="books", user=user, password=password
)

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
    
def split_large_context(doc_content, max_length=120000):  # Keep some buffer for query
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, chunk_overlap=200  # Prevent breaking sentences
    )
    return text_splitter.split_text(doc_content)

# Function to process a single chunk
def process_chunk(chunk, query):
    system_prompt = """
    You are an intelligent assistant. Use the given context to answer the question as accurately as possible.
    If the answer is not explicitly in the context, make an educated guess.
    The answer should always be in the below-mentioned format. Never change the format of the answer.

    Format the answer based on the query:
    - If the question requires a single response, return the answer in this format:
    {{
        "answer": "string",
         "article_id": "string",
        "article_name": "string",
        "category_id": "string",
        "category_name": "string"
    }}
    - If the question requires a list of objects, return the answer as an array:
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
    ]

    If you cannot answer the question based on the context, respond with:
    {{
        "answer": "I don't know",
        "article_id": "",
        "article_name": "",
        "category_id": "",
        "category_name": ""
    }}

    Context:
    {context}

    Question: {query}
    """
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    return chain.invoke({"query": query, "context": chunk})

# Function to process each chunk and merge the results
def process_multiple_chunks(doc_content, query):
    context_chunks = split_large_context(doc_content)

    # Use ThreadPoolExecutor to process all chunks concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda chunk: process_chunk(chunk, query), context_chunks))

    merged_answers = [result.text if hasattr(result, 'text') else str(result) for result in results]
    return merged_answers
    
def query_articles(article_id: str, query: str):
    conn = db_pool.getconn()
    try:
        if conn is None:
            return {"error": "Unable to acquire a connection from the pool."}, 500
        cur = conn.cursor()

        print("Connected to the database")

        # Generate embedding for the query
        query_embedding = sentence_transformer_ef.encode(query)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Define a similarity threshold (you can adjust this based on testing)
        similarity_threshold = 0.5

        # Perform vector similarity search with a threshold using pgvector
        sql_query = """
            SELECT text_content, embedding_vector
            FROM articles
            WHERE article_id = %s
              AND embedding_vector IS NOT NULL
              AND embedding_vector <=> %s::vector < %s  -- Pre-filter based on similarity
            ORDER BY embedding_vector <=> %s::vector
        """
        cur.execute(sql_query, (article_id, query_embedding.tolist(), similarity_threshold, query_embedding.tolist()))
        rows = cur.fetchall()

        # Handle no results
        if not rows:
            return {"error": "No relevant documents found."}, 404
        
        print(f"Retrieved {len(rows)} documents from db")

        # Process results efficiently
        document_texts = []
        doc_embeddings = []
        for row in rows:
            document_texts.append(row[0])
            doc_embeddings.append(np.array(json.loads(row[1]), dtype=np.float32))
        
        # Compute similarity using scipy
        similarities = 1 - cdist([query_embedding], doc_embeddings, metric="cosine").flatten()
        
        # Filter results by similarity threshold
        top_documents = [
            {"page_content": document_texts[i], "metadata": {"similarity": similarities[i]}}
            for i in range(len(similarities)) if similarities[i] > similarity_threshold
        ]

        print(f"Retrieved {len(top_documents)} relevant documents")

        # Sort documents by similarity and prepare them
        top_documents.sort(key=lambda x: x["metadata"]["similarity"], reverse=True)

        # Combine document contents for context
        doc_content = "\n".join([doc["page_content"] for doc in top_documents])
        doc_content = doc_content.replace("{", "").replace("}", "")

        # Process the query in chunks
        answers = process_multiple_chunks(doc_content, query)
        print(f"Answer Length: {len(answers)}")
        valid_answers = []

        for item in answers:
            match_1 = re.search(r'```json\\n({.*?})\\n```', item, re.DOTALL)  # Single JSON object
            match_2 = re.search(r'json\\n(\[.*\])\\n', item, re.DOTALL)  # JSON array

            json_str = None  # Initialize json_str

            if match_1:
                json_str = match_1.group(1)
            elif match_2:
                json_str = match_2.group(1)

            if json_str and not re.search(r'"answer":\s*"I don\'t know"', json_str):
                valid_answers.append(json_str)

        print(f"Answer Options Length: {len(valid_answers)}")

        # Prepare the prompt for the AI to choose final answer
        ai_prompt="""You are an intelligent assistant. From the given context, choose the one that best answers the query.
        If the answer is not explicitly in the context, make an educated guess.
        Context: {context}
        Query: {query}.

        The answer should always be in the below-mentioned format.

        Format the answer based on the query:
        - If the question requires a single response, return the answer in this format:
        {{
            "answer": "string",
            "article_id": "string",
            "article_name": "string",
            "category_id": "string",
            "category_name": "string"
        }}
        - If the question requires a list of objects, return the answer as an array:
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
        ]
        """
        prompt = PromptTemplate.from_template(ai_prompt)
        chain = prompt | llm
        response = chain.invoke({"context": valid_answers, "query": query})
        if hasattr(response, 'text'):  # Check if it's an object with 'text' attribute
            final_answer = response.text
        else:
            # If it's already a string, you can use it directly
            final_answer = str(response)

        match = re.search(r"content='(.*?)' response_metadata", final_answer, re.DOTALL)

        if match:
            answer_data = match.group(1)
        else:
            print("No valid JSON found in the answer")

        # Parse the answer to JSON
        try:
            # Remove ```json and ``` if present
            cleaned_answer = answer_data.replace("```json", "").replace("```", "").strip()
            cleaned_answer = cleaned_answer.replace("\\n", "").strip()  # Remove newline escape characters
            cleaned_answer = cleaned_answer.replace("'", "\"").strip()  # Replace single quotes with double quotes

            # Ensure JSON ends properly
            if not cleaned_answer.endswith("]") and "[" in cleaned_answer:
                cleaned_answer += "]"  # Attempt to close the array if missing
            
            # Parse the JSON string into a Python object (could be a dict or list)
            parsed_answer = json.loads(cleaned_answer)
            print(f"Parsed Answer: {parsed_answer}")
            
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
            raise ValueError(f"Invalid JSON format in the answer: {cleaned_answer}. Error: {e}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while processing the answer: {e}")

        return {"answer": response, "answer_list": response_list, "context": top_documents[:5]}, 200

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return {"error": str(e)}, 500

    finally:
        cur.close()
        db_pool.putconn(conn)