import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document
from app.langchain.components.file_extractor import extract_text_from_file
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from fastapi import UploadFile, HTTPException
from typing import List
import re
import requests
import psycopg2
from sqlalchemy import delete as sa_delete, text


os.environ["TOKENIZERS_PARALLELISM"] = "false"

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

load_dotenv()

_PG_DSN = f"postgresql://{user}:{password}@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books"

llm = init_chat_model("gpt-4o", model_provider="openai")
fast_llm = init_chat_model("gpt-4o", model_provider="openai")
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


def _delete_by_metadata(key: str, value: str):
    """
    Delete all vector store documents whose metadata contains key=value.
    Uses SQLAlchemy directly since langchain_postgres 0.0.17 delete() only
    supports deletion by id list.
    """
    try:
        with vector_store._make_sync_session() as session:
            collection = vector_store.get_collection(session)
            if not collection:
                print(f"[motanabi] Warning: Collection not found, skipping delete for {key}={value}")
                return
            stmt = (
                sa_delete(vector_store.EmbeddingStore)
                .where(vector_store.EmbeddingStore.collection_id == collection.uuid)
                .where(
                    vector_store.EmbeddingStore.cmetadata[key].astext == value
                )
            )
            result = session.execute(stmt)
            session.commit()
            print(f"[motanabi] Deleted {result.rowcount} existing chunk(s) where {key}='{value}'.")
    except Exception as e:
        print(f"[motanabi] Warning: Could not delete existing docs for {key}='{value}': {e}")


def store_file(files: List[UploadFile]):
    try:
        all_documents = []
        succeeded = []
        failed = []

        for file in files:
            # Extract text from each uploaded file
            extracted_text = extract_text_from_file(file)
            if not extracted_text:
                print(f"[motanabi] ❌ No text extracted from '{file.filename}'. Skipping.")
                failed.append({"file": file.filename, "reason": "No text could be extracted"})
                continue

            # Replace strategy: delete existing docs for this filename before re-adding
            _delete_by_metadata("original_filename", file.filename)

            # Split documents/pages into chunks, preserving metadata
            chunks = text_splitter.split_documents(extracted_text)

            # Strip NUL bytes (\x00) that Postgres rejects in string columns
            # (common in binary-mixed or corrupted PDFs)
            for chunk in chunks:
                chunk.page_content = chunk.page_content.replace('\x00', '').strip()
                # Tag with original filename for future deduplication
                chunk.metadata["original_filename"] = file.filename

            # Drop any chunks that are empty after sanitization
            chunks = [c for c in chunks if c.page_content]

            print(f"[motanabi] Extracted {len(chunks)} chunks from '{file.filename}'.")
            all_documents.extend(chunks)
            succeeded.append({"file": file.filename, "chunks": len(chunks)})

        if not all_documents:
            print("\n[motanabi] ===== Upload Summary =====")
            for f in failed:
                print(f"  ❌ '{f['file']}' — {f['reason']}")
            print("[motanabi] =========================\n")
            return {"error": "No text could be extracted from any of the provided files."}, 400

        # Store all documents from all files in a single call
        print(f"[motanabi] Storing {len(all_documents)} total chunks in the motanabi vector store.")
        _ = vector_store.add_documents(documents=all_documents)

        # Console upload summary
        print("\n[motanabi] ===== Upload Summary =====")
        for s in succeeded:
            print(f"  ✅ '{s['file']}' — {s['chunks']} chunks stored successfully.")
        for f in failed:
            print(f"  ❌ '{f['file']}' — {f['reason']}")
        print("[motanabi] =========================\n")

        return {"status": "success"}, 200

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}, 500


def store_text_content(text: str, source_name: str, extra_metadata: dict = None):
    """
    Store raw string text directly into the motanabi vector store.
    Deletes existing documents with the same source_name first (replace strategy).
    """
    try:
        # Replace strategy: delete existing docs for this source
        _delete_by_metadata("source", source_name)

        metadata = {"source": source_name}
        if extra_metadata:
            metadata.update(extra_metadata)

        doc = Document(page_content=text, metadata=metadata)
        chunks = text_splitter.split_documents([doc])

        # Sanitize
        for chunk in chunks:
            chunk.page_content = chunk.page_content.replace('\x00', '').strip()

        chunks = [c for c in chunks if c.page_content]

        if not chunks:
            return {"error": "No usable text content after processing."}, 400

        print(f"[motanabi] Storing {len(chunks)} chunks for source '{source_name}'.")
        vector_store.add_documents(documents=chunks)

        return {"status": "success", "chunks_stored": len(chunks)}, 200
    except Exception as e:
        return {"error": str(e)}, 500


def _augment_query_for_retrieval(question: str) -> str:
    """
    Convert a question (in Arabic OR English) into a bilingual (English + Arabic)
    retrieval query so the multilingual embedding model can find the most relevant
    Arabic poem chunks regardless of the question language.

    Examples:
        "bring the poem lines that talk about the horses"
        → "horses خيل جياد أفراس فرس Al-Mutanabbi poems about horses"

        "في أي سنة ماتت أم سيف الدولة وما هي القصيدة التي رثاها المتنبي بها"
        → "وفاة أم سيف الدولة رثاء المتنبي قصيدة رثاء death elegy Al-Mutanabbi"
    """
    prompt = (
        "You are a bilingual Arabic-English search query specialist.\n"
        "The question below may be written in Arabic OR in English — handle both equally.\n"
        "Your task: output a SHORT bilingual retrieval query that includes:\n"
        "  1. If the question mentions a SPECIFIC POEM TITLE (in Arabic, transliterated, or in English), "
        "include the Arabic title as the first priority.\n"
        "  2. The key topic and constraint keywords in BOTH Arabic and English "
        "(especially temporal/order constraints like 'first/أول', 'last/آخر', 'death/وفاة', 'birth/ولادة', "
        "'elegy/رثاء', 'praise/مدح', 'most famous/أشهر').\n"
        "  3. Arabic synonyms and related words for the topic (space-separated).\n"
        "  4. A short descriptive Arabic phrase like 'رثاء المتنبي لـ[person]' or 'قصيدة المتنبي عن [topic]'.\n"
        "Output ONLY the query string, no explanation.\n\n"
        f"Question: {question}\n"
        "Bilingual retrieval query:"
    )
    response = llm.invoke(prompt)
    augmented = response.content.strip()
    print(f"[motanabi] Augmented retrieval query: {augmented!r}")
    return augmented if augmented else question


def _prepare_search_inputs(question: str, conversation_history: str = None) -> tuple[str, list[str]]:
    """
    Consolidates question rewriting, retrieval query augmentation, and Arabic keyword extraction
    into a single LLM call to minimize API latency.
    
    Returns:
        tuple[retrieval_query, list_of_arabic_keywords]
    """
    if conversation_history:
        history_context = (
            f"Conversation History:\n{conversation_history}\n\n"
            f"Follow-up Question: {question}\n"
        )
        instruction = (
            "First, resolve the follow-up question using the conversation history to determine the "
            "standalone topic the user is asking about. Then perform the tasks below based on that resolved topic."
        )
    else:
        history_context = f"Question: {question}\n"
        instruction = "Perform the tasks below based on this question."

    prompt = (
        "You are an AI assistant specialized in search query optimization and Classical Arabic poetry (specifically Al-Mutanabbi's diwan).\n"
        f"{history_context}\n"
        f"{instruction}\n\n"
        "Your task is to generate search inputs for our retrieval pipeline. Output a JSON object with exactly two keys:\n"
        "1. \"retrieval_query\": A SHORT bilingual (English + Arabic) search query optimized for a multilingual vector store. "
        "Include the key topic, Al-Mutanabbi, and synonyms in both languages (e.g. 'horses خيل جياد'). "
        "If a specific poem title or recipient is mentioned, prioritize the Arabic name/title.\n"
        "2. \"arabic_keywords\": A JSON array of key Arabic words, phrases, and expanded synonyms for direct database filtering (SQL ILIKE). "
        "Rules for keywords:\n"
        "  - Include the Arabic form of any mentioned poem titles, recipients, or event locations.\n"
        "  - If the input is a QUOTED VERSE or partial verse (e.g. 'أنبكي لموتانا على غير رغبة'), include:\n"
        "      * The quoted verse phrase itself (e.g. 'أنبكي لموتانا').\n"
        "      * Each individual significant word from the verse (e.g. 'نبكي', 'موتانا', 'رغبة').\n"
        "      * Synonyms and morphological variants: e.g., 'نبكي' → also 'أبكي', 'بكاء', 'يبكي'; 'موتانا'/'موتى' → also 'أموات', 'الموتى'; 'رغبة' → also 'إرادة'.\n"
        "  - Expand common synonyms (e.g., if 'أم' is present add 'والدة'; if 'وفاة'/'مات' add 'توفي', 'ماتت', 'وفاتها'; if 'رثاء' add 'يرثي', 'رثاها', 'مرثية').\n"
        "  - Include specific multi-word search phrases representing relationships or events asked about (e.g. 'والدة سيف الدولة', 'وفاة أم سيف الدولة', 'أول قصيدة مدح بها سيف الدولة'). This is critical for matching book headers.\n"
        "  - Keep individual terms clean and relevant.\n\n"
        "Respond ONLY with a valid JSON object. Do not include markdown code block formatting or explanations. Example output format:\n"
        "{\n"
        "  \"retrieval_query\": \"horses Al-Mutanabbi خيل جياد أفراس فرس\",\n"
        "  \"arabic_keywords\": [\"خيل\", \"جياد\", \"أفراس\", \"فرس\"]\n"
        "}"
    )

    try:
        response = fast_llm.invoke(prompt)
        content = response.content.strip()
        # strip markdown code blocks if the LLM outputted them
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n|```$", "", content, flags=re.IGNORECASE).strip()
        
        import json
        data = json.loads(content)
        retrieval_query = data.get("retrieval_query", "").strip() or question
        arabic_keywords = data.get("arabic_keywords", [])
        
        print(f"[motanabi] Optimized retrieval query: {retrieval_query!r}")
        print(f"[motanabi] Optimized Arabic keywords: {arabic_keywords}")
        return retrieval_query, [k.strip() for k in arabic_keywords if k.strip()]
    except Exception as e:
        print(f"[motanabi] Warning: Consolidated query preparation failed: {e}. Falling back to default behavior.")
        # Fallback to simple query and empty keywords
        return question, []


def query_motanabi(question: str):
    return _query_motanabi_core(question)


def query_motanabi_with_context(question: str, conversation_history: str):
    return _query_motanabi_core(question, conversation_history)


def _extract_arabic_keywords(question: str) -> list[str]:
    """
    Ask the LLM to extract key Arabic words/phrases from a question written
    in Arabic OR English (including poem titles in transliterated form).
    Returns a list of Arabic strings to use as SQL ILIKE search terms.
    """
    prompt = (
        "You are an Arabic-English specialist for Al-Mutanabbi poetry.\n"
        "The question below may be written in Arabic OR in English — handle both equally.\n"
        "Extract from the question:\n"
        "  1. Any poem title mentioned (return it in its Arabic form, even if given as transliteration or in English).\n"
        "  2. Key Arabic content words, synonyms, or phrases relevant to the question.\n"
        "  3. If a verse is quoted or translated, include both the phrase AND its individual core words and synonyms "
        "(e.g., if 'weep' / 'نبكي' is mentioned, include 'نبكي', 'أبكي', 'بكاء'; if 'dead' / 'موتى' is mentioned, include 'موتانا', 'موتى', 'أموات').\n"
        "  4. Temporal/order/superlative constraints in Arabic (e.g., 'first' / 'أول' -> 'أول', 'أقدم'; 'last' / 'آخر' -> 'آخر'; 'most famous' / 'أشهر' -> 'أشهر'; 'death' / 'وفاة' -> 'وفاة', 'مات', 'توفي').\n"
        "  5. SYNONYM EXPANSION — always expand these common synonyms:\n"
        "     - 'أم' (mother) → also include 'والدة' (they are synonyms; the source text may use either).\n"
        "     - 'رثى / رثاء' (elegy) → also include 'يرثي', 'مرثية', 'رثاها'.\n"
        "     - 'مات / وفاة' (death) → also include 'توفي', 'ماتت', 'وفاتها'.\n"
        "  6. If the question asks about a specific relationship, event, person, or fact, extract a full descriptive Arabic search phrase for it "
        "(e.g., 'والدة سيف الدولة', 'وفاة أم سيف الدولة', 'رثاء المتنبي لأم سيف الدولة', 'أول قصيدة مدح بها سيف الدولة'). This is highly critical for matching book passages.\n"
        "Output ONLY a JSON array of Arabic strings (no explanation), e.g.:\n"
        '  ["أم سيف الدولة", "والدة سيف الدولة", "وفاة", "رثاء", "مات", "توفي", "يرثي"]\n\n'
        f"Question: {question}\n"
        "Arabic keywords JSON array:"
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        # Extract JSON array from response
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            import json
            keywords = json.loads(m.group())
            keywords = [k.strip() for k in keywords if k.strip()]
            print(f"[motanabi] Arabic keywords for SQL search: {keywords}")
            return keywords
    except Exception as e:
        print(f"[motanabi] Warning: keyword extraction failed: {e}")
    return []


def _sql_keyword_search(keywords: list[str], limit: int = 10) -> list[Document]:
    """
    Perform a direct PostgreSQL ILIKE search on the motanabi collection
    for any of the given Arabic keywords. Chunks are ranked by the number
    of matched keywords and ordered deterministically.
    """
    if not keywords:
        return []

    # Exclude common stop words/broad search terms that match almost all documents
    stop_words = {
        'مدح', 'شعر', 'قصيدة', 'ديوان', 'المتنبي', 'أبو الطيب', 'موضوع', 'سيف الدولة',
        'كافور', 'ممدوح', 'ابن', 'في', 'من', 'على', 'عن', 'إلى', 'مع', 'أو', 'أن', 'لا'
    }

    # Hardcoded Arabic synonym expansions — deterministic, no LLM dependency.
    # If a word appears in extracted keywords, its synonyms are also searched.
    SYNONYM_MAP: dict[str, list[str]] = {
        'أم':      ['والدة'],
        'والدة':   ['أم'],
        'رثاء':    ['يرثي', 'مرثية', 'رثاها'],
        'يرثي':    ['رثاء', 'مرثية', 'رثاها'],
        'مرثية':   ['رثاء', 'يرثي'],
        'رثاها':   ['رثاء', 'يرثي', 'مرثية'],
        'وفاة':    ['ماتت', 'توفي', 'توفيت', 'مات'],
        'ماتت':    ['وفاة', 'توفيت'],
        'مات':     ['وفاة', 'توفي'],
        'توفي':    ['وفاة', 'ماتت', 'توفيت'],
        'توفيت':   ['وفاة', 'ماتت'],
        'أخت':     ['أخته'],
        'أخته':    ['أخت'],
        # Verse-completion synonyms
        'نبكي':    ['أبكي', 'يبكي', 'بكاء', 'نبكي'],
        'أبكي':    ['نبكي', 'يبكي', 'بكاء'],
        'يبكي':    ['نبكي', 'أبكي', 'بكاء'],
        'بكاء':    ['نبكي', 'أبكي', 'يبكي'],
        'موتانا':  ['موتى', 'أموات', 'الموتى'],
        'موتى':    ['موتانا', 'أموات', 'الموتى'],
        'أموات':   ['موتانا', 'موتى', 'الموتى'],
        'الموتى':  ['موتانا', 'موتى', 'أموات'],
        'رغبة':    ['إرادة', 'مشيئة'],
        'إرادة':   ['رغبة', 'مشيئة'],
    }

    expanded_keywords = []
    for kw in keywords:
        kw_clean = kw.strip()
        if kw_clean and len(kw_clean) > 1 and kw_clean not in stop_words:
            expanded_keywords.append(kw_clean)
            # Apply synonym expansion for individual words found in this multi-word keyword
            for word in re.findall(r'\w+', kw_clean):
                for syn in SYNONYM_MAP.get(word, []):
                    expanded_keywords.append(syn)
        # Also split multi-word phrases into individual words
        for w in re.findall(r'\w+', kw):
            w_clean = w.strip()
            if w_clean and len(w_clean) > 1 and w_clean not in stop_words:
                expanded_keywords.append(w_clean)
                for syn in SYNONYM_MAP.get(w_clean, []):
                    expanded_keywords.append(syn)

    # Deduplicate while preserving order
    seen = set()
    filtered_keywords = []
    for kw in expanded_keywords:
        if kw not in seen:
            filtered_keywords.append(kw)
            seen.add(kw)

    if not filtered_keywords:
        return []

    try:
        score_parts = []
        conditions = []
        params = {}
        
        for i, kw in enumerate(filtered_keywords):
            param_name = f"kw_{i}"
            score_parts.append(f"(CASE WHEN e.document ILIKE :{param_name} THEN 1 ELSE 0 END)")
            conditions.append(f"e.document ILIKE :{param_name}")
            params[param_name] = f"%{kw}%"
            
        score_expr = " + ".join(score_parts)
        where_expr = " OR ".join(conditions)
        params["limit_val"] = limit * 3
        
        sql_query = text(f"""
            SELECT e.document, e.cmetadata, ({score_expr}) as match_score
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = 'motanabi'
              AND ({where_expr})
            ORDER BY match_score DESC, e.cmetadata->>'source' ASC, e.id ASC
            LIMIT :limit_val;
        """)
        
        docs = []
        seen_contents = set()
        session = None
        try:
            with vector_store._make_sync_session() as s:
                session = s
                result = session.execute(sql_query, params)
                for row in result:
                    if len(docs) >= limit:
                        break
                    content = row[0]
                    content_key = content[:100]
                    if content_key not in seen_contents:
                        docs.append(Document(page_content=content, metadata=row[1] or {}))
                        seen_contents.add(content_key)
        finally:
            if session:
                session.close()
                    
        print(f"[motanabi] SQL keyword search returned {len(docs)} extra chunk(s) using keywords: {filtered_keywords}")
        return docs
    except Exception as e:
        print(f"[motanabi] Warning: SQL keyword search failed: {e}")
        return []


def _sql_poem_header_search(keywords: list[str], limit: int = 5) -> list[Document]:
    """
    Search ONLY in poemsTxtFile source chunks using multi-word phrase keywords.
    Poem header chunks contain the structured description: poemId, title,
    occasion/year the poem was composed, and the opening lines.
    These are the most authoritative source for factual questions (year, title, etc.).

    We ONLY use multi-word phrases (e.g. "والدة سيف الدولة") rather than single words
    (e.g. "وفاة") because single words match hundreds of poem headers and introduce
    noise. Multi-word phrases are specific enough to pinpoint the exact poem.
    Results are prepended first in the LLM context so factual details are seen first.
    """
    if not keywords:
        return []

    # Prefer multi-word phrases (contain a space) as they are far more specific.
    # Fall back to long single words (>4 chars) only if no phrases are available.
    phrase_terms = [kw.strip() for kw in keywords if ' ' in kw.strip() and len(kw.strip()) > 3]
    
    # Automatically expand 'أم' <-> 'والدة' in phrase search terms to ensure robust matching
    expanded_phrases = []
    for p in phrase_terms:
        if p not in expanded_phrases:
            expanded_phrases.append(p)
        if 'أم' in p:
            alt = p.replace('أم', 'والدة')
            if alt not in expanded_phrases:
                expanded_phrases.append(alt)
        if 'والدة' in p:
            alt = p.replace('والدة', 'أم')
            if alt not in expanded_phrases:
                expanded_phrases.append(alt)

    fallback_terms = [kw.strip() for kw in keywords if ' ' not in kw.strip() and len(kw.strip()) > 4]

    search_terms = expanded_phrases if expanded_phrases else fallback_terms
    if not search_terms:
        return []

    try:
        params = {}
        conditions = []
        for i, kw in enumerate(search_terms):
            param_name = f"kw_{i}"
            conditions.append(f"e.document ILIKE :{param_name}")
            params[param_name] = f"%{kw}%"
        
        conditions_str = " OR ".join(conditions)
        # Fetch more candidates because we filter by source type in Python
        params["limit_val"] = limit * 6

        sql_query = text(f"""
            SELECT e.document, e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = 'motanabi'
              AND ({conditions_str})
            ORDER BY e.id ASC
            LIMIT :limit_val;
        """)

        docs = []
        seen = set()
        session = None
        try:
            with vector_store._make_sync_session() as s:
                session = s
                result = session.execute(sql_query, params)
                for row in result:
                    if len(docs) >= limit:
                        break
                    content = row[0]
                    metadata = row[1] or {}
                    # Python filtering of source type to avoid PostgreSQL unpacking TOAST data (JSONB) on full table scan
                    if metadata.get("source", "").startswith("poemsTxtFile"):
                        key = content[:100]
                        if key not in seen:
                            docs.append(Document(page_content=content, metadata=metadata))
                            seen.add(key)
        finally:
            if session:
                session.close()

        print(f"[motanabi] Poem header search returned {len(docs)} chunk(s) from poemsTxtFile "
              f"(using {'phrases' if phrase_terms else 'keywords'}: {search_terms[:3]})")
        return docs
    except Exception as e:
        print(f"[motanabi] Warning: Poem header search failed: {e}")
        return []


def _query_motanabi_core(question: str, conversation_history: str = None):
    try:
        # ── Step 1: Build retrieval query and extract keywords in a consolidated LLM call ──
        retrieval_query, arabic_keywords = _prepare_search_inputs(question, conversation_history)

        # ── Step 2: Retrieve documents using the enriched query ─────────────────
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 30}
        )
        source_documents = retriever.invoke(retrieval_query)

        # ── Step 2b: SQL keyword fallback ────────────────────────────────────────
        # For poem-title / metadata questions, embedding search can miss the right
        # chunk due to spelling variants (عواذل vs أواذل) or transliterations.
        # Use direct ILIKE search using consolidated keywords, then merge.
        if arabic_keywords:
            # Step 2b-i: Poem header search (HIGHEST PRIORITY)
            # poemsTxtFile chunks contain structured year/title/description — they
            # are the single most authoritative source for factual questions.
            # Insert them first so the LLM sees them before any other content.
            poem_header_docs = _sql_poem_header_search(arabic_keywords, limit=3)
            existing_contents = {d.page_content[:100] for d in source_documents}
            for doc in reversed(poem_header_docs):  # reversed so first result ends up at index 0
                if doc.page_content[:100] not in existing_contents:
                    source_documents.insert(0, doc)
                    existing_contents.add(doc.page_content[:100])

            # Step 2b-ii: Regular SQL keyword fallback (alwaraq + all sources)
            sql_docs = _sql_keyword_search(arabic_keywords, limit=5)
            for doc in sql_docs:
                if doc.page_content[:100] not in existing_contents:
                    source_documents.insert(len(poem_header_docs), doc)  # after poem headers
                    existing_contents.add(doc.page_content[:100])

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

        IMPORTANT — Language rule (apply ALWAYS, for every response type):
        - Detect the language of the Question below.
        - If the Question is written in Arabic, you MUST respond entirely in Arabic for all parts of your answer (greetings, explanations, poem lines, everything).
        - If the Question is written in English, respond in English (poem lines may still be quoted in their original Arabic).
        - Never mix the response language; be consistent throughout.

        First, decide what type of question this is (do NOT output the type/category prefix letter like "A)", "B)", "C)" or "Type A:" in your final response - just output the response content itself):
        A) GREETING / SMALL TALK (e.g. "Hello", "Hi", "How are you?", "Thank you", or their Arabic equivalents) —
           Respond warmly and naturally in the detected language (Arabic if the question is in Arabic, English otherwise). Do NOT mention poems or verses.
        B) FACTUAL QUESTIONS ABOUT AL-MUTANABBI, HIS POEMS, OR SPECIFIC LINES/VERSES
           (e.g., biography, style, completing or explaining a specific verse, identifying who is praised in a specific poem or line, etc.) —
           1. Scan ALL retrieved material carefully for the answer. Answer directly and concisely in the detected language using what you find. Facts may be stated in Arabic prose, in Arabic numerals, or as Arabic number-words (e.g., "سبع وثلاثين وثلاثمائة" = 337 AH) — read them all and summarize accurately.
           2. VERSE COMPLETION — if the question quotes a verse (in Arabic OR English) and asks to complete it (e.g., "هل يمكنك إكمال البيت", "complete the verse", "what comes next"):
              a. If the verse is quoted in ARABIC (e.g., "أنبكي لموتانا على غير رغبة"): search the retrieved chunks for that exact phrase or its key words (نبكي, موتانا, رغبة), locate the full couplet in the source, and output the completing hemistich or next line exactly as written.
              b. If the verse is quoted in ENGLISH TRANSLATION (e.g., "We weep for our dead against our will"): translate it mentally to the matching Arabic line, then output the next line or lines in Arabic (copying exactly as written).
              In BOTH cases, copy the lines exactly as they appear in the retrieved material — do NOT paraphrase or modify.
           3. If the question is in English but refers to events, relationships, or first poems (e.g., "first poem praising Sayf al-Dawla"), scan the Arabic material for relevant mentions (e.g., references to first contact "أول اتصاله به" or first recitation/poetry "أول ما أنشده" / "أول شعر") and answer the question in English, quoting the poem's opening line (e.g., "وفاؤكما كالربع أشجاه طاسمه") or title in Arabic.
           4. Do NOT mention the word "context" (such as "provided context", "according to the context", "based on the context", etc.) anywhere in your response. Simply state the answer directly based on the facts.
           5. No hallucination rule: Only report facts that are present in the retrieved material. Do NOT invent, guess, or estimate specific dates, years, poem titles, verses, or names that are not found anywhere in the retrieved material. If after scanning ALL retrieved material the specific fact is genuinely absent, say so clearly in the detected language (e.g., "لم تُذكر هذه المعلومة في المادة المتوفرة." or "This information was not found in the available material.") — but only say this as a last resort after a thorough scan.
           6. PRECISION rule — match the EXACT person/subject asked about: The retrieved material may contain facts about MULTIPLE people (e.g., Sayf al-Dawla's mother أم AND his sister أخت, or different poets, or different events). You MUST match the date, poem, or fact ONLY to the specific person or subject the question is asking about. For example, if the question asks about the MOTHER (أم / والدة), use ONLY dates and poems from chunks that explicitly describe the mother — ignore any dates or poems in chunks about the sister (أخت) or any other person.
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
        6. If you cannot find any matching lines after scanning all excerpts, respond in the detected language only (Arabic if the question is in Arabic, English otherwise):
             "No related verses were found in the collection for this topic." (or its Arabic equivalent: "لم يُعثر على أبيات ذات صلة بهذا الموضوع في المجموعة.")
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


def fetch_book_pages(book_id: str, language: str):
    """
    Fetch book pages from the Alwaraq API, print the result, then store
    the fullBookPages text into the motanabi vector store.
    Existing chunks for the same bookId are replaced (deleted then re-added).
    """
    url = f"https://alwaraq.net/json_bookallpages.php?language={language}&bookId={book_id}"
    print(f"[motanabi] Fetching book pages from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse JSON or fallback to raw text
        try:
            data = response.json()
        except ValueError:
            data = response.text

        print("[motanabi] Raw API result:")
        print(data)

        # ── Store fullBookPages into vector store ────────────────────────────
        full_text = data.get("fullBookPages", "") if isinstance(data, dict) else ""
        book_name = data.get("name", f"book_{book_id}") if isinstance(data, dict) else f"book_{book_id}"
        author = data.get("author", "unknown") if isinstance(data, dict) else "unknown"

        print("\n[motanabi] ===== Book Pages Upload Summary =====")
        if full_text:
            source_name = f"alwaraq_book_{book_id}"
            store_result, store_status = store_text_content(
                text=full_text,
                source_name=source_name,
                extra_metadata={
                    "book_id": book_id,
                    "book_name": book_name,
                    "author": author,
                    "language": language,
                }
            )
            if store_status == 200:
                chunks = store_result.get("chunks_stored", "?")
                print(f"  ✅ '{book_name}' (bookId={book_id}) — {chunks} chunks stored successfully.")
            else:
                error_msg = store_result.get("error", "Unknown error")
                print(f"  ❌ '{book_name}' (bookId={book_id}) — Failed to store: {error_msg}")
        else:
            print(f"  ⚠️  No fullBookPages content found for bookId={book_id}. Nothing stored.")
        print("[motanabi] ==========================================\n")

        return {"status": "success", "data": data}, 200
    except Exception as e:
        print(f"\n[motanabi] ===== Book Pages Upload Summary =====")
        print(f"  ❌ bookId={book_id} — Fetch failed: {e}")
        print("[motanabi] ==========================================\n")
        return {"error": str(e)}, 500

