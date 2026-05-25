import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from app.langchain.components.file_extractor import extract_text_from_file
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from fastapi import UploadFile, HTTPException
from typing import List
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


def store_file(files: List[UploadFile]):
    try:
        all_documents = []

        for file in files:
            # Extract text from each uploaded file
            extracted_text = extract_text_from_file(file)
            if not extracted_text:
                print(f"Warning: No text extracted from file '{file.filename}'. Skipping.")
                continue

            # Split documents/pages into chunks, preserving metadata
            chunks = text_splitter.split_documents(extracted_text)

            # Strip NUL bytes (\x00) that Postgres rejects in string columns
            # (common in binary-mixed or corrupted PDFs)
            for chunk in chunks:
                chunk.page_content = chunk.page_content.replace('\x00', '').strip()

            # Drop any chunks that are empty after sanitization
            chunks = [c for c in chunks if c.page_content]

            print(f"Extracted {len(chunks)} chunks from '{file.filename}'.")
            all_documents.extend(chunks)

        if not all_documents:
            return {"error": "No text could be extracted from any of the provided files."}, 400

        # Store all documents from all files in a single call
        print(f"Storing {len(all_documents)} total documents in the motanabi vector store.")
        _ = vector_store.add_documents(documents=all_documents)

        return {"status": "success"}, 200

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}, 500


def query_motanabi(question: str):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 30}
        )

        prompt_template = """
        You are a Conversational AI assistant specializing in the poetry of Al-Mutanabbi.
        Only use the information from the context to answer the question.
        If no related verses are found in the context, respond politely in English,
        explaining that no related verses were found in the collection.
        Do NOT make up or hallucinate poem lines.

        Critical Instructions (follow exactly):
        - Return poem lines that are semantically or thematically related to the subject asked about in the question.
          This includes lines that directly mention the subject AND lines whose meaning or imagery is clearly about the topic.
        - Do NOT mix information from different poems.
        - You MUST reference the poemId for every line you mention using this EXACT format: [poemId: ID]
          Example: "وَقَفتُ عَلى رَبعٍ لِمَيَّةَ" [poemId: 42]
          NEVER use any other format such as "PoemId: 42" or "(poemId: 42)" — brackets are mandatory.
        - If no lines in the context are related to the subject, respond in English only,
          e.g. "No related verses were found in the collection for this topic."

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


def query_motanabi_with_context(question: str, conversation_history: str):
    """
    Memory-aware variant of query_motanabi.

    When conversation history is present, the follow-up question is first
    rewritten into a self-contained standalone query so the vector retriever
    can find the right documents even for vague follow-ups like
    "any other poems?" or "tell me more".

    Args:
        question:             The user's current question.
        conversation_history: Pre-formatted history string (may be empty)

    Returns:
        (dict, int) — same shape as query_motanabi().
    """
    try:
        # ── Step 1: Rewrite follow-up into standalone retrieval query ────────
        # Only triggered when there is prior conversation context.
        retrieval_query = question
        if conversation_history:
            retrieval_query = _rewrite_standalone_question(
                question, conversation_history
            )
            print(f"[motanabi] Rewritten retrieval query: {retrieval_query!r}")

        # ── Step 2: Retrieve documents using the enriched query ──────────────
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 30}
        )
        source_docs = retriever.invoke(retrieval_query)

        # ── Step 3: Build context string from retrieved docs ─────────────────
        context_text = "\n\n".join(doc.page_content for doc in source_docs)

        # ── Step 4: Build the final prompt (history + context + question) ────
        history_section = (
            f"\n        Conversation History:\n        {conversation_history}\n"
            if conversation_history
            else ""
        )

        prompt_template = f"""
        You are a Conversational AI assistant specializing in the poetry of Al-Mutanabbi.
        Only use the information from the context to answer the question.
        If no related verses are found in the context, respond politely in English,
        explaining that no related verses were found in the collection.
        Do NOT make up or hallucinate poem lines.{history_section}

        Critical Instructions (follow exactly):
        - Return poem lines that are semantically or thematically related to the subject asked about in the question.
          This includes lines that directly mention the subject AND lines whose meaning or imagery is clearly about the topic.
        - Do NOT mix information from different poems.
        - You MUST reference the poemId for every line you mention using this EXACT format: [poemId: ID]
          Example: "وَقَفتُ عَلى رَبعٍ لِمَيَّةَ" [poemId: 42]
          NEVER use any other format such as "PoemId: 42" or "(poemId: 42)" — brackets are mandatory.
        - If no lines in the context are related to the subject, respond in English only,
          e.g. "No related verses were found in the collection for this topic."
        - If the conversation history already contains verses on this topic, you MAY return
          additional different verses from the context. Do NOT repeat poem lines that were
          already shown in the conversation history.

        Context:
        {{context}}

        Question:
        {{question}}
        """

        chat_prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        # ── Step 5: LLM inference with pre-retrieved docs ────────────────────
        # We pass the original question to the LLM (not the rewritten one)
        # so the answer reads naturally.
        chain = chat_prompt | llm
        response_msg = chain.invoke({"context": context_text, "question": question})
        answer = response_msg.content

        # ── Step 6: Extract and clean poemIds ────────────────────────────────
        poem_ids = re.findall(r"\[poemId:\s*(\w+)\]", answer, re.IGNORECASE)
        cleaned_answer = re.sub(
            r"\[poemId:\s*\w+\]", "", answer, flags=re.IGNORECASE
        ).strip()

        return {
            "question": question,
            "answer": cleaned_answer,
            "poemIds": list(set(poem_ids)),
            "context": source_docs,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


def _rewrite_standalone_question(question: str, conversation_history: str) -> str:
    """
    Reformulate a vague follow-up question into a self-contained topic query
    suitable for vector similarity search.

    Rules:
      - Expand the topic so the retriever can find relevant documents.
      - NEVER add "excluding poemId X" — that would filter out the only
        matching documents and cause empty results.
      - Repetition-avoidance is handled by the LLM in the answer prompt,
        NOT at the retrieval stage.

    Example:
        history:  User asked about horses, got poemIds 71, 88
        question: "any other poems?"
        →         "Al-Mutanabbi poems about horses"
    """
    prompt = (
        "You are a search-query rewriting assistant.\n"
        "Given a conversation history and a follow-up question, rewrite the follow-up "
        "as a standalone topic query for a vector similarity search.\n\n"
        "Rules:\n"
        "- Focus on the TOPIC the user is asking about.\n"
        "- Do NOT include phrases like 'excluding', 'other than', or 'besides poemId X'.\n"
        "- Output ONLY the rewritten query, nothing else.\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"Follow-up Question: {question}\n\n"
        "Standalone Topic Query:"
    )
    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    # Fallback: if LLM returns empty, use original question
    return rewritten if rewritten else question
