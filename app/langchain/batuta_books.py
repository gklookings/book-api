import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from psycopg2 import extras
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
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

def store_batuta_documents(trip_id: str, files, text: str):
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
                
        # If data exists for this tripId, delete it
        cur.execute('DELETE FROM trips WHERE tripid = %s', (trip_id,))
        conn.commit()
        print("Old data removed if it existed.")

        # Insert new data
        query = """
            INSERT INTO trips (tripid, text_content, embedding_vector)
            VALUES %s
        """
        values = [
            (trip_id, texts[i], np.array(embeddings[i], dtype=np.float32).tolist()) for i in range(len(texts))
        ]

        try:
            extras.execute_values(cur, query, values, page_size=batch_size)
            conn.commit()
            print("Data inserted successfully")
        finally:
            cur.close()
            conn.close()

        return {"trip_id": trip_id, "status": "success"}, 200

    except Exception as e:
        return {"error": str(e)}, 500
    
def query_batuta_documents(trip_id: str, query: str):
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

        # Generate embedding for the query
        query_embedding = sentence_transformer_ef.encode(query)  # Convert the query to an embedding

        # Convert query_embedding to a numpy array of type np.float32
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Perform vector similarity search using pgvector
        sql_query = f"""
        SELECT embedding_vector, text_content
        FROM trips
        WHERE tripid = %s
        ORDER BY embedding_vector <=> %s::vector  -- Explicitly cast to vector
        LIMIT 150;
        """
        
        # Execute the query with adapted query embedding (passing it as a numpy array directly)
        cur.execute(sql_query, (trip_id, query_embedding.tolist()))
        rows = cur.fetchall()

        # Ensure there are results
        if not rows:
            return {"error": "No relevant documents found."}, 404

        # Extract embeddings and text content from the fetched rows
        doc_embeddings = [np.array(ast.literal_eval(row[0]), dtype=np.float32) for row in rows]
        document_texts = [row[1] for row in rows]

        # Sort by similarity and get the top results
        top_documents = [
            {
                "page_content": document_texts[i],
                "metadata": {
                    "similarity": 1 - np.dot(query_embedding, doc_embeddings[i]) / 
                    (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embeddings[i]))
                }
            }
            for i in range(len(doc_embeddings))
        ]
        print(f"Retrieved {len(top_documents)} relevant documents")

        # Extract only the page content for LangChain
        document_contents = [doc["page_content"] for doc in top_documents]
        documents = [Document(page_content=document_texts[i]) for i in range(len(document_contents))]

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
        answer = question_answer_chain.invoke({"context": documents, "input": query})

        # Close the cursor and release the connection back to the pool
        cur.close()
        conn.close()

        return {"answer": answer, "context": top_documents[:5]}, 200

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return {"error": str(e)}, 500