"""
Science Museum quiz — live API client + translation.

Nothing is stored: every quiz request fetches a fresh random set of scientists
from the Science Museum API (isQuiz=1) in the requested language. Names and
descriptions only come in that language, so the ones a quiz uses are
translated into the other language with gpt-4o-mini (UAE MSA for Arabic).

The field list (json_fields.php) is tiny and rarely changes, so it is kept in
process memory for FIELDS_TTL_SECONDS instead of being fetched per request.
"""
import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

SCIENTISTS_API_URL = "https://sciencemuseum.electronicvillage.org/json_museum.php"
FIELDS_API_URL = "https://sciencemuseum.electronicvillage.org/json_fields.php"
API_LANGUAGES = {"ar": 1, "en": 2}

REQUEST_TIMEOUT = 60
REQUEST_ATTEMPTS = 3
FIELDS_TTL_SECONDS = 6 * 3600
DESCRIPTION_LIMIT = 2000
TRANSLATE_BATCH_SIZE = 60
TRANSLATE_WORKERS = 4

# The API spells Nobel categories inconsistently ("physics", "Chemistry",
# "medicine" and "Physiology or Medicine" for the same prize).
NOBEL_ALIASES = {
    "physics": "physics",
    "chemistry": "chemistry",
    "medicine": "medicine",
    "physiology or medicine": "medicine",
    "literature": "literature",
    "peace": "peace",
    "economic sciences": "economics",
    "economics": "economics",
}

translator_llm = init_chat_model(
    "gpt-4o-mini", model_provider="openai", temperature=0
).bind(response_format={"type": "json_object"})

_TRANSLATE_PROMPTS = {
    "ar": (
        "You translate quiz content from English into Arabic as written in UAE media "
        "(Modern Standard Arabic). The input is a JSON object. Keys starting with \"n\" "
        "are names of people (scientists, writers, Nobel laureates): transliterate them "
        "the way UAE Arabic-language media (e.g. Al Khaleej, Al Bayan, Emarat Al Youm) "
        "writes them; Arab figures get their original Arabic spelling. Keys starting with "
        "\"t\" are short texts: translate them faithfully, keep every \"…\" placeholder "
        "exactly where it is and never add a name. Reply with a JSON object mapping the "
        "same keys to the translation only."
    ),
    "en": (
        "You translate quiz content from Arabic into English. The input is a JSON object. "
        "Keys starting with \"n\" are names of people (scientists, writers, Nobel "
        "laureates): write them in the standard English spelling used by Wikipedia "
        "(e.g. ماري كوري -> Marie Curie, ابن سينا -> Ibn Sina). Keys starting with \"t\" "
        "are short texts: translate them faithfully, keep every \"…\" placeholder exactly "
        "where it is and never add a name. Reply with a JSON object mapping the same keys "
        "to the translation only."
    ),
}


class ScienceMuseumApiError(RuntimeError):
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────


def _clean_name(value) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _clean_text(value) -> str:
    """Strip HTML, Nobel section headings and placeholders from API descriptions."""
    if not value:
        return ""
    cleaned = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    # Nobel write-ups carry "<h3>Nobel Prize — Physics (1905)</h3>" and
    # "<strong>The idea</strong>" style labels; both would give the answer away or add noise.
    cleaned = re.sub(r"<(h\d|strong)\b[^>]*>.*?</\1>", " ", cleaned, flags=re.S | re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned).replace("\x00", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.fullmatch(r"x*", cleaned, flags=re.I):
        return ""
    return cleaned[:DESCRIPTION_LIMIT]


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("؟ "), cut.rfind("? "))
    if sentence_end > limit * 0.5:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def _parse_year(value):
    """'1867-11-07' -> 1867, '0965-07-01' -> 965; '0000-00-00' / None -> None."""
    match = re.match(r"(\d{1,4})-", value or "")
    year = int(match.group(1)) if match else 0
    return year if 0 < year <= time.gmtime().tm_year else None


def _nobels(details) -> list:
    entries, seen = [], set()
    for detail in details or []:
        category = NOBEL_ALIASES.get(str(detail.get("field") or "").strip().lower())
        try:
            year = int(detail.get("year"))
        except (TypeError, ValueError):
            continue
        if not category or (year, category) in seen:
            continue
        seen.add((year, category))
        citation = re.sub(r"<[^>]+>", "", html.unescape(detail.get("citation") or ""))
        entries.append({"year": year, "category": category, "citation": citation.strip()})
    return entries


def _scientist(article):
    """Normalise one API article; None for entries that aren't people."""
    name = _clean_name(article.get("name"))
    birth_year = _parse_year(article.get("birthDate"))
    picture = article.get("coverImage")
    # Topic articles ("Mont Blanc.. the secret of Earth's geology", "Quantum Computing")
    # sit in the same list; they have no birth date or a ".." title.
    if not name or not picture or not birth_year or ".." in name:
        return None
    return {
        "id": str(article.get("articleId")),
        "name": name,
        "picture": picture,
        "birthYear": birth_year,
        "description": _clean_text(article.get("description")),
        "nobels": _nobels(article.get("nobelDetails")),
        "fieldId": None,
    }


def _post(url: str, data: dict) -> dict:
    last_error = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return json.loads(response.content.decode("utf-8-sig").strip())
        except (requests.RequestException, ValueError) as e:
            last_error = e
            time.sleep(attempt + 1)
    raise ScienceMuseumApiError(f"Science Museum API request failed: {last_error}")


# ── Public API ───────────────────────────────────────────────────────────────


def fetch_scientists(language: str, count: int, field_id: int | None = None) -> list:
    """A fresh random list of scientists (people only, unique ids)."""
    data = {"language": API_LANGUAGES[language], "isQuiz": 1, "interval": count}
    if field_id:
        data["fieldId"] = field_id
    scientists = {}
    for article in _post(SCIENTISTS_API_URL, data).get("articles") or []:
        scientist = _scientist(article)
        if scientist:
            scientist["fieldId"] = field_id
            scientists.setdefault(scientist["id"], scientist)
    return list(scientists.values())


_fields_lock = threading.Lock()
_fields_cache = {"fields": None, "loaded_at": 0.0}


def get_fields() -> dict:
    """{fieldId: {"en": name, "ar": name, "picture": url}}; {} if the API is down."""
    with _fields_lock:
        if _fields_cache["fields"] and time.time() - _fields_cache["loaded_at"] < FIELDS_TTL_SECONDS:
            return _fields_cache["fields"]
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                en, ar = pool.map(
                    lambda lang: _post(FIELDS_API_URL, {"language": API_LANGUAGES[lang]}),
                    ("en", "ar"),
                )
        except ScienceMuseumApiError as e:
            print(f"[science_museum_quiz] Warning: fields unavailable: {e}")
            return _fields_cache["fields"] or {}
        ar_names = {str(f.get("fieldId")): _clean_name(f.get("name")) for f in ar.get("articles") or []}
        fields = {}
        for field in en.get("articles") or []:
            key = str(field.get("fieldId"))
            name_en, name_ar = _clean_name(field.get("name")), ar_names.get(key)
            if key.isdigit() and name_en and name_ar:
                fields[int(key)] = {"en": name_en, "ar": name_ar, "picture": field.get("coverImage")}
        _fields_cache.update(fields=fields, loaded_at=time.time())
        return fields


def _translate_batch(batch, target: str) -> dict:
    payload = {f"{'n' if kind == 'name' else 't'}{i}": text for i, (kind, text) in enumerate(batch)}
    try:
        response = translator_llm.invoke(
            [
                ("system", _TRANSLATE_PROMPTS[target]),
                ("human", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        data = json.loads(response.content)
    except Exception as e:
        print(f"[science_museum_quiz] Warning: translation batch failed: {e}")
        return {}
    result = {}
    for i, (kind, text) in enumerate(batch):
        value = data.get(f"{'n' if kind == 'name' else 't'}{i}")
        if isinstance(value, str) and value.strip():
            result[(kind, text)] = value.strip()
    return result


def translate(items, source_language: str) -> dict:
    """Translate {(kind, text)} from the source language into the other one.

    kind is "name" (transliterate) or "text" (translate). Returns
    {(kind, text): translation}; anything that fails is simply missing.
    """
    target = "en" if source_language == "ar" else "ar"
    items = sorted(set(items))
    batches = [items[i : i + TRANSLATE_BATCH_SIZE] for i in range(0, len(items), TRANSLATE_BATCH_SIZE)]
    result = {}
    if not batches:
        return result
    with ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS) as pool:
        for part in pool.map(lambda batch: _translate_batch(batch, target), batches):
            result.update(part)
    print(f"[science_museum_quiz] Translated {len(result)}/{len(items)} items to {target}.")
    return result
