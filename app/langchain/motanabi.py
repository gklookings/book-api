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


def _augment_query_for_retrieval(question: str) -> str:
    """
    Translate an English question into a bilingual (English + Arabic) retrieval
    query so the multilingual embedding model can find the most relevant Arabic
    poem chunks.

    Example:
        "bring the poem lines that talk about the horses"
        → "horses خيل جياد أفراس فرس Al-Mutanabbi poems about horses"
    """
    prompt = (
        "You are a bilingual Arabic-English search query specialist.\n"
        "Given an English question about Al-Mutanabbi poetry, output a SHORT retrieval "
        "query that includes:\n"
        "  1. The key topic in English\n"
        "  2. The Arabic words / synonyms for that topic (space-separated)\n"
        "  3. A short Arabic phrase like 'شعر المتنبي عن [topic]'\n"
        "Output ONLY the query string, no explanation.\n\n"
        f"English question: {question}\n"
        "Bilingual retrieval query:"
    )
    response = llm.invoke(prompt)
    augmented = response.content.strip()
    print(f"[motanabi] Augmented retrieval query: {augmented!r}")
    return augmented if augmented else question


def query_motanabi(question: str):
    return _query_motanabi_core(question)


def query_motanabi_with_context(question: str, conversation_history: str):
    return _query_motanabi_core(question, conversation_history)


def _query_motanabi_core(question: str, conversation_history: str = None):
    try:
        # ── Step 1: Build retrieval query ───────────────────────────────────────
        if conversation_history:
            # Rewrite follow-up as standalone, then augment with Arabic
            standalone = _rewrite_standalone_question(question, conversation_history)
            print(f"[motanabi] Rewritten retrieval query: {standalone!r}")
            retrieval_query = _augment_query_for_retrieval(standalone)
        else:
            retrieval_query = _augment_query_for_retrieval(question)

        # ── Step 2: Retrieve documents using the enriched query ─────────────────
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 30}
        )
        source_documents = retriever.invoke(retrieval_query)

        # ── Step 3: Build context string from retrieved docs ──────────────────
        # Build context: extract poemId from source filename (e.g. mot149.txt → 149)
        # and prepend as a guaranteed header so the LLM always knows the poemId,
        # even for continuation chunks that lack the "poemId is  X" header text.
        def _pid_from_doc(doc) -> str:
            src = doc.metadata.get("source", "")
            m = re.search(r"mot(\d+)", src)
            return m.group(1) if m else "unknown"

        context_text = "\n\n---\n\n".join(
            f"[POEM ID: {_pid_from_doc(doc)}]\n{doc.page_content}"
            for doc in source_documents
        )

        # ── Step 4: Build the final prompt (history + context + question) ────────
        history_section = (
            f"\n        Conversation History:\n        {conversation_history}\n"
            if conversation_history
            else ""
        )

        prompt_template = f"""
        You are an expert in Classical Arabic poetry, specialising in Al-Mutanabbi's diwan.
        You are also a friendly, conversational assistant.
        The context below contains excerpts from Al-Mutanabbi's poems written in Classical Arabic.
        Each excerpt starts with [POEM ID: X] — this is the poemId for ALL lines in that excerpt.
        {{history_section}}

        First, decide what type of question this is (do NOT output the type/category prefix letter like "A)", "B)", "C)" or "Type A:" in your final response - just output the response content itself):
        A) GREETING / SMALL TALK (e.g. "Hello", "Hi", "How are you?", "Thank you") —
           Respond warmly and naturally in English. Do NOT mention poems or verses.
        B) FACTUAL QUESTION ABOUT AL-MUTANABBI (birth, death, biography, style, etc.) —
           Answer directly and concisely using information from the context.
           Do NOT start with "According to the provided context" or any similar preamble.
        C) POEM/VERSE SEARCH (asking for poem lines about a topic) —
           Follow the rules below to find and return matching lines.

        Rules for TYPE C only (must follow exactly):
        1. Scan every numbered line (e.g. "1 | ...", "2 | ...") in every excerpt in the context.
        2. Identify the Arabic word(s) for the topic. For example:
             - "horses" → look for: خيل، جياد، أفراس، فرس، طرف، مهر
             - "sword"  → look for: سيف، صارم، حسام
             - "sea"    → look for: بحر، يم، موج
           Do the same Arabic-word lookup for any other topic.
        3. Return a line if and only if:
             a) It contains one of those Arabic words, OR
             b) Its entire meaning is unmistakably and primarily about the topic.
           Do NOT return a line just because it is in the same poem.
        4. For EVERY matching line, tag it using the [POEM ID: X] of the excerpt it came from:
             "<line text>" [poemId: X]
           NEVER write [poemId: Unknown]. The [POEM ID: X] header is ALWAYS present and gives you the ID.
           NEVER omit the tag. NEVER use any other bracket style.
        5. Do NOT repeat lines that already appear in the Conversation History (if provided above).
        6. If you cannot find any matching lines after scanning all excerpts, respond in English only:
             "No related verses were found in the collection for this topic."
        7. Do NOT make up or modify any poem lines. Copy them exactly as written.

        Context:
        {{context}}

        Question:
        {{question}}
        """

        # Format history_section dynamic token
        formatted_prompt = prompt_template.format(
            history_section=history_section,
            context="{context}",
            question="{question}"
        )

        chat_prompt = PromptTemplate(
            input_variables=["context", "question"], template=formatted_prompt
        )

        # ── Step 5: LLM inference ──────────────────────────────────────────
        chain = chat_prompt | llm
        response_msg = chain.invoke({"context": context_text, "question": question})
        answer = response_msg.content

        # ── Step 6: Extract and clean poemIds ────────────────────────────────
        # Extract poemIds, filter out any "unknown" that slipped through
        poem_ids = re.findall(r"\[poemId:\s*(\w+)\]", answer, re.IGNORECASE)
        poem_ids = [pid for pid in poem_ids if pid.lower() != "unknown"]
        cleaned_answer = re.sub(r"\[poemId:\s*\w+\]", "", answer, flags=re.IGNORECASE).strip()

        # Remove any category prefix like "A) ", "B) ", "Category A:", "Type A - " etc.
        cleaned_answer = re.sub(
            r"^(?:Type\s+[A-C]\s*[-:]*|[A-C]\s*[-)]\s*|Category\s+[A-C]:\s*)",
            "",
            cleaned_answer,
            flags=re.IGNORECASE
        ).strip()

        return {
            "question": question,
            "answer": cleaned_answer,
            "poemIds": list(set(poem_ids)),
            "context": source_documents,
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
