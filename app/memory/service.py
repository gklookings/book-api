"""
app/memory/service.py

Memory business logic:
  - Load memory window for a session+domain
  - Build the conversation-history string injected into prompts
  - Save a Q+A exchange back to memory (rolling window + summarisation)

Rolling window rule (per ARCHITECTURE.md):
  - Keep max 10 messages (5 Q+A pairs) in recent_messages
  - When > 10: remove the oldest 2 (1 pair) and run LLM summarisation
"""

from typing import Optional
from langchain.chat_models import init_chat_model

from app.memory import repository

# ── Lazy LLM singleton for summarisation ────────────────────────────────────
# Instantiated on first use so that load_dotenv() has already run by then.
_summariser_llm = None


def _get_summariser_llm():
    global _summariser_llm
    if _summariser_llm is None:
        _summariser_llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    return _summariser_llm

_MAX_MESSAGES = 10  # rolling window cap


# ── Public API ────────────────────────────────────────────────────────────────


def load_context(session_token: str, domain: str) -> str:
    """
    Return a formatted conversation-history string ready to inject into a
    RAG prompt.  Returns an empty string when there is no prior memory.

    Format:
        [Summary of earlier conversation]
        User: ...
        Assistant: ...
        User: ...
        Assistant: ...
    """
    if not session_token:
        return ""

    memory = repository.get_memory(session_token, domain)
    if memory is None:
        return ""

    parts: list[str] = []

    if memory.get("summary"):
        parts.append(f"[Summary of earlier conversation]\n{memory['summary']}")

    for msg in memory.get("recent_messages", []):
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")

    return "\n".join(parts)


def save_exchange(
    session_token: str,
    domain: str,
    question: str,
    answer: str,
) -> None:
    """
    Persist a single Q+A exchange:
      1. Append to rolling window (recent_messages)
      2. If window > MAX_MESSAGES → drop oldest pair + LLM summarise
      3. UPSERT user_memory
      4. INSERT two rows into chat_history
    """
    if not session_token:
        return

    # Load existing memory (or start fresh)
    memory = repository.get_memory(session_token, domain) or {
        "summary": None,
        "recent_messages": [],
        "message_count": 0,
    }

    recent: list[dict] = memory.get("recent_messages", [])
    summary: Optional[str] = memory.get("summary")
    message_count: int = memory.get("message_count", 0)

    # Append new pair
    recent.append({"role": "user", "content": question})
    recent.append({"role": "assistant", "content": answer})
    message_count += 2

    # Rolling window: trim + summarise when over the limit
    if len(recent) > _MAX_MESSAGES:
        # Pop the oldest Q+A pair (2 messages)
        evicted_user = recent.pop(0)
        evicted_assistant = recent.pop(0)

        summary = _summarise(
            existing_summary=summary,
            question=evicted_user["content"],
            answer=evicted_assistant["content"],
        )

    # Persist memory
    repository.upsert_memory(
        session_token=session_token,
        domain=domain,
        recent_messages=recent,
        message_count=message_count,
        summary=summary,
    )

    # Append to full history log
    repository.append_chat_history(session_token, domain, "user", question)
    repository.append_chat_history(session_token, domain, "assistant", answer)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _summarise(
    existing_summary: Optional[str],
    question: str,
    answer: str,
) -> str:
    """
    Ask the LLM to compress an evicted Q+A pair into the running summary.
    """
    llm = _get_summariser_llm()
    prompt = (
        "You are a memory compression assistant.\n"
        "Summarize the following conversation exchange in 1–2 concise sentences "
        "to preserve key context for future interactions.\n\n"
        f"Existing summary: {existing_summary or 'None'}\n\n"
        f"New exchange:\n"
        f"User: {question}\n"
        f"Assistant: {answer}\n\n"
        "Return only the updated summary text."
    )
    response = llm.invoke(prompt)
    return response.content.strip()
