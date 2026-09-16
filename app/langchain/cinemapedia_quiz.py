"""
Cinemapedia quiz — vector store + ingestion.

Collection: "cinemapedia_quiz" (PGVector, same Postgres instance as the other modules).

Document types (metadata["doc_type"]), ids are uuid5 of these keys:
  - "film"      one per movie        key = "film:<movieId>"
  - "artist"    one per actor/director key = "artist:<artistId>"
  - "glossary"  single document       key = "glossary:terms"
                (Arabic translations of countries, festivals and award names)

Artist names only exist in English in the source API, so Arabic spellings are
generated once at ingestion time (gpt-4o-mini) and stored in the artist
metadata. Quiz generation therefore never calls an LLM.
"""
import os
import re
import json
import html
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from statistics import median

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

COLLECTION_NAME = "cinemapedia_quiz"
MOVIES_API_URL = "https://cinemapediaapi.electronicvillage.org/moviesforQuiz.php"
SESSION_TOKEN = os.getenv("CINEMAPEDIA_SESSION_TOKEN", "ccfca260f7d628ec1")

DIRECTOR_TYPE_ID = 2
ACTOR_TYPE_ID = 5
ACTRESS_TYPE_ID = 6
ACTOR_TYPE_IDS = {ACTOR_TYPE_ID, ACTRESS_TYPE_ID}

DESCRIPTION_LIMIT = 400
UPSERT_BATCH_SIZE = 200
TRANSLATE_BATCH_SIZE = 150
PICTURE_CHECK_WORKERS = 24
EXCLUDED_COUNTRIES = {"", "unknown"}

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=f"postgresql+psycopg2://{user}:{password}@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books",
)

translator_llm = init_chat_model(
    "gpt-4o-mini", model_provider="openai", temperature=0
).bind(response_format={"type": "json_object"})

_TRANSLATE_PROMPTS = {
    "names": (
        "You transliterate the names of film artists (actors, actresses, directors) "
        "into Arabic script, spelled the way UAE Arabic-language media (e.g. Al Khaleej, "
        "Al Bayan, Emarat Al Youm) writes them. Arab artists must get their original "
        "Arabic spelling. The input is a JSON object mapping ids to names. Reply with a "
        "JSON object mapping the same ids to the Arabic name only."
    ),
    "countries": (
        "You translate country names into Arabic as written in UAE official media "
        "(Modern Standard Arabic). The input is a JSON object mapping ids to country "
        "names. Reply with a JSON object mapping the same ids to the Arabic name only."
    ),
    "festivals": (
        "You translate names of film festivals and award ceremonies into Arabic as "
        "written in UAE media (e.g. Oscars -> جوائز الأوسكار, Cannes Film Festival -> "
        "مهرجان كان السينمائي). The input is a JSON object mapping ids to names. Reply "
        "with a JSON object mapping the same ids to the Arabic name only."
    ),
    "awards": (
        "You translate film award category names into Arabic as written in UAE media "
        "(e.g. Best Picture -> جائزة أفضل فيلم, Best Actress -> جائزة أفضل ممثلة). The "
        "input is a JSON object mapping ids to award names. Reply with a JSON object "
        "mapping the same ids to the Arabic name only."
    ),
}

# Bumped after every successful sync / clean so the quiz generator reloads its cache.
data_version = 0

_sync_lock = threading.Lock()
sync_state = {
    "running": False,
    "startedAt": None,
    "finishedAt": None,
    "currentPage": None,
    "filmsStored": 0,
    "artistsStored": 0,
    "error": None,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def has_arabic(value) -> bool:
    return bool(value and ARABIC_RE.search(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_title(value) -> str:
    """Unescape quotes and drop a stray trailing full stop ("Girl." -> "Girl", keeps "...")."""
    cleaned = (value or "").replace("\\'", "'").replace('\\"', '"')
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return re.sub(r"(?<!\.)\s*\.$", "", cleaned)


def _clean_text(value, limit: int = DESCRIPTION_LIMIT) -> str:
    """Strip HTML / escaped line breaks from API descriptions and truncate."""
    if not value:
        return ""
    cleaned = html.unescape(value)
    cleaned = (
        cleaned.replace("\\r", " ")
        .replace("\\n", " ")
        .replace('\\"', '"')
        .replace("\\'", "'")
        .replace("\x00", "")
    )
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("؟ "))
    if sentence_end > limit * 0.5:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def _parse_year(value):
    match = re.match(r"(\d{4})", value or "")
    year = int(match.group(1)) if match else None
    return year if year and 1880 <= year <= 2100 else None


def _fetch_page(page: int) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                MOVIES_API_URL,
                params={"page": page},
                headers={"sessiontoken": SESSION_TOKEN},
                timeout=90,
            )
            response.raise_for_status()
            # The API prefixes its JSON with a UTF-8 BOM and whitespace.
            return json.loads(response.content.decode("utf-8-sig").strip())
        except (requests.RequestException, ValueError) as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch page {page}: {last_error}")


def _is_valid_picture(url: str) -> bool:
    """Cinemapedia returns HTTP 200 with an empty body for missing pictures."""
    if not url:
        return False
    try:
        with requests.get(url, timeout=20, stream=True) as response:
            if response.status_code != 200:
                return False
            if not response.headers.get("content-type", "").startswith("image/"):
                return False
            first_chunk = next(response.iter_content(1024), b"")
            return len(first_chunk) > 0
    except requests.RequestException:
        return False


def _check_pictures(urls) -> dict:
    urls = list({u for u in urls if u})
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=PICTURE_CHECK_WORKERS) as pool:
        return dict(zip(urls, pool.map(_is_valid_picture, urls)))


def _translate_terms(terms, kind: str) -> dict:
    """Translate / transliterate a list of terms to Arabic. Returns {term: arabic}."""
    result = {}
    terms = sorted({t for t in terms if t})
    for start in range(0, len(terms), TRANSLATE_BATCH_SIZE):
        batch = terms[start : start + TRANSLATE_BATCH_SIZE]
        payload = {str(i): term for i, term in enumerate(batch)}
        try:
            response = translator_llm.invoke(
                [
                    ("system", _TRANSLATE_PROMPTS[kind]),
                    ("human", json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = json.loads(response.content)
            for key, value in data.items():
                if key in payload and isinstance(value, str) and value.strip():
                    result[payload[key]] = value.strip()
        except Exception as e:
            print(f"[cinemapedia_quiz] Warning: {kind} translation batch failed: {e}")
    print(f"[cinemapedia_quiz] Translated {len(result)}/{len(terms)} {kind}.")
    return result


def _doc_id(key: str) -> str:
    """The embedding id column is a UUID, so map our readable keys to stable UUIDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION_NAME}/{key}"))


def _upsert(documents, keys):
    ids = [_doc_id(k) for k in keys]
    for start in range(0, len(documents), UPSERT_BATCH_SIZE):
        vector_store.add_documents(
            documents=documents[start : start + UPSERT_BATCH_SIZE],
            ids=ids[start : start + UPSERT_BATCH_SIZE],
        )


# ── Read access (used by the quiz generator) ─────────────────────────────────


def load_metadata(doc_type: str) -> list:
    """Return the metadata of every document of the given type in the collection."""
    with vector_store._make_sync_session() as session:
        rows = session.execute(
            text(
                """
                SELECT e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :collection
                  AND e.cmetadata->>'doc_type' = :doc_type
                """
            ),
            {"collection": COLLECTION_NAME, "doc_type": doc_type},
        ).fetchall()
    return [row[0] for row in rows]


def similar_film_ids(query: str, k: int = 12) -> list:
    """Movie ids of the films most similar to the query (used for distractors)."""
    try:
        docs = vector_store.similarity_search(
            query, k=k, filter={"doc_type": {"$eq": "film"}}
        )
        return [d.metadata.get("movie_id") for d in docs]
    except Exception as e:
        print(f"[cinemapedia_quiz] Warning: similarity search failed: {e}")
        return []


# ── Films ────────────────────────────────────────────────────────────────────


def _film_metadata(entity: dict, poster_valid: bool) -> dict:
    name_ar_field = clean_title(entity.get("name_1"))
    name_en_field = clean_title(entity.get("name_2"))
    title_en = name_en_field or name_ar_field
    title_ar = name_ar_field if has_arabic(name_ar_field) else None

    description_ar = _clean_text(entity.get("description_1"))
    description_en = _clean_text(entity.get("description_2"))
    if not has_arabic(description_ar):
        description_ar = ""

    countries = []
    for country in entity.get("countries") or []:
        name = (country.get("name_2") or country.get("name_1") or "").strip()
        if name.lower() not in EXCLUDED_COUNTRIES and name not in countries:
            countries.append(name)

    crew = [
        {
            "artistId": c.get("artistId"),
            "crewTypeId": c.get("crewTypeId"),
            "crewName": c.get("crewName"),
            "artistName": clean_title(c.get("artistName")),
            "picture": c.get("picture"),
        }
        for c in entity.get("crew") or []
        if c.get("artistId") and (c.get("artistName") or "").strip()
    ]

    def _awards(items):
        return [
            {
                "festival": (n.get("festivalName") or "").strip(),
                "award": (n.get("awardName") or "").strip(),
                "year": str(n.get("year") or "").strip(),
            }
            for n in items or []
            if (n.get("festivalName") or "").strip() and (n.get("awardName") or "").strip()
        ]

    return {
        "doc_type": "film",
        "movie_id": entity["id"],
        "title_en": title_en,
        "title_ar": title_ar,
        "description_en": description_en,
        "description_ar": description_ar,
        "year": _parse_year(entity.get("releaseDate")),
        "picture": entity.get("picture"),
        "has_picture": poster_valid,
        "countries": countries,
        "crew": crew,
        "nominations": _awards(entity.get("nominations")),
        "winners": _awards(entity.get("winners")),
        "has_arabic": bool(title_ar and description_ar),
    }


def _film_document(meta: dict) -> Document:
    directors = [c["artistName"] for c in meta["crew"] if c["crewTypeId"] == DIRECTOR_TYPE_ID]
    cast = [c["artistName"] for c in meta["crew"] if c["crewTypeId"] in ACTOR_TYPE_IDS]
    awards = [f"{a['award']} ({a['festival']} {a['year']})".strip() for a in meta["winners"]]
    lines = [
        f"Film: {meta['title_en']}" + (f" | {meta['title_ar']}" if meta["title_ar"] else ""),
        f"Year: {meta['year'] or 'unknown'}",
        f"Countries: {', '.join(meta['countries'])}" if meta["countries"] else "",
        f"Directors: {', '.join(directors)}" if directors else "",
        f"Cast: {', '.join(cast)}" if cast else "",
        f"Awards won: {', '.join(awards)}" if awards else "",
        meta["description_en"],
        meta["description_ar"],
    ]
    return Document(
        page_content="\n".join(line for line in lines if line),
        metadata=meta,
    )


def _store_film_page(entities: list) -> int:
    validity = _check_pictures(e.get("picture") for e in entities)
    metas = [_film_metadata(e, validity.get(e.get("picture"), False)) for e in entities if e.get("id")]
    _upsert([_film_document(m) for m in metas], [f"film:{m['movie_id']}" for m in metas])
    return len(metas)


# ── Artists + glossary ───────────────────────────────────────────────────────


def _rebuild_artists_and_glossary() -> int:
    """
    Rebuild artist documents and the glossary from every stored film, so a
    partial page sync never leaves an artist with an incomplete filmography.
    Existing Arabic names / translations and valid pictures are reused.
    """
    films = load_metadata("film")
    titles = {f["movie_id"]: f["title_en"] for f in films}
    existing_artists = {a["artist_id"]: a for a in load_metadata("artist")}
    existing_glossary = next(iter(load_metadata("glossary")), {})

    artists = {}
    for film in films:
        for c in film.get("crew", []):
            type_id = c["crewTypeId"]
            if type_id != DIRECTOR_TYPE_ID and type_id not in ACTOR_TYPE_IDS:
                continue
            artist = artists.setdefault(
                c["artistId"],
                {
                    "name_en": c["artistName"],
                    "picture": c["picture"],
                    "type_ids": set(),
                    "acted_in": set(),
                    "directed": set(),
                    "years": [],
                    "arabic_films": False,
                },
            )
            artist["type_ids"].add(type_id)
            if type_id == DIRECTOR_TYPE_ID:
                artist["directed"].add(film["movie_id"])
            else:
                artist["acted_in"].add(film["movie_id"])
            if film.get("year"):
                artist["years"].append(film["year"])
            if film.get("has_arabic"):
                artist["arabic_films"] = True

    # Pictures: only re-check the ones not already known to be valid.
    known_valid = {
        a["picture"]
        for a in existing_artists.values()
        if a.get("has_picture") and a.get("picture")
    }
    validity = _check_pictures(
        a["picture"] for a in artists.values() if a["picture"] not in known_valid
    )

    # Arabic names: reuse when the English name is unchanged.
    arabic_names = {}
    for artist_id, artist in artists.items():
        previous = existing_artists.get(artist_id)
        if previous and previous.get("name_en") == artist["name_en"] and previous.get("name_ar"):
            arabic_names[artist["name_en"]] = previous["name_ar"]
    missing = [a["name_en"] for a in artists.values() if a["name_en"] not in arabic_names]
    arabic_names.update(_translate_terms(missing, "names"))

    documents, keys = [], []
    for artist_id, artist in artists.items():
        type_ids = artist["type_ids"]
        roles = []
        if type_ids & ACTOR_TYPE_IDS:
            roles.append("actor")
        if DIRECTOR_TYPE_ID in type_ids:
            roles.append("director")
        gender = None
        if ACTRESS_TYPE_ID in type_ids and ACTOR_TYPE_ID not in type_ids:
            gender = "female"
        elif ACTOR_TYPE_ID in type_ids and ACTRESS_TYPE_ID not in type_ids:
            gender = "male"
        name_ar = arabic_names.get(artist["name_en"])
        film_ids = sorted(artist["acted_in"] | artist["directed"])
        meta = {
            "doc_type": "artist",
            "artist_id": artist_id,
            "name_en": artist["name_en"],
            "name_ar": name_ar,
            "picture": artist["picture"],
            "has_picture": artist["picture"] in known_valid
            or validity.get(artist["picture"], False),
            "roles": roles,
            "gender": gender,
            "acted_in": sorted(artist["acted_in"]),
            "directed": sorted(artist["directed"]),
            "year": int(median(artist["years"])) if artist["years"] else None,
            "has_arabic": bool(name_ar and artist["arabic_films"]),
        }
        film_titles = ", ".join(titles[i] for i in film_ids[:30] if titles.get(i))
        content = (
            f"Artist: {meta['name_en']}"
            + (f" | {name_ar}" if name_ar else "")
            + f"\nRoles: {', '.join(roles)}\nFilms: {film_titles}"
        )
        documents.append(Document(page_content=content, metadata=meta))
        keys.append(f"artist:{artist_id}")

    _upsert(documents, keys)

    stale = [f"artist:{i}" for i in existing_artists if i not in artists]
    if stale:
        vector_store.delete(ids=[_doc_id(k) for k in stale])
        print(f"[cinemapedia_quiz] Removed {len(stale)} stale artist(s).")

    # Glossary: Arabic for countries, festivals and award names.
    glossary = {"doc_type": "glossary"}
    sources = {
        "countries": {c for f in films for c in f.get("countries", [])},
        "festivals": {a["festival"] for f in films for a in f.get("nominations", []) + f.get("winners", [])},
        "awards": {a["award"] for f in films for a in f.get("nominations", []) + f.get("winners", [])},
    }
    for kind, terms in sources.items():
        previous = existing_glossary.get(kind) or {}
        translated = {t: previous[t] for t in terms if previous.get(t)}
        translated.update(_translate_terms([t for t in terms if t not in translated], kind))
        glossary[kind] = translated
    _upsert(
        [Document(page_content="Cinemapedia glossary (English to Arabic terms)", metadata=glossary)],
        ["glossary:terms"],
    )

    return len(documents)


# ── Public entry points ──────────────────────────────────────────────────────


def sync_movies(start_page: int = 1, end_page: int = None):
    """
    Fetch movies page by page from the Cinemapedia API and upsert them into the
    vector store, then rebuild artists + glossary. Intended to run in a
    background thread; progress is exposed through `sync_state`.
    """
    global data_version
    if not _sync_lock.acquire(blocking=False):
        print("[cinemapedia_quiz] A sync is already running — skipping.")
        return
    sync_state.update(
        running=True, startedAt=_now(), finishedAt=None, currentPage=None,
        filmsStored=0, artistsStored=0, error=None,
    )
    try:
        page = start_page
        while end_page is None or page <= end_page:
            sync_state["currentPage"] = page
            data = _fetch_page(page)
            entities = data.get("entities") or []
            if entities:
                sync_state["filmsStored"] += _store_film_page(entities)
                print(f"[cinemapedia_quiz] Page {page}: {len(entities)} film(s) stored (total {sync_state['filmsStored']}).")
            if data.get("isLastPage") or not entities:
                break
            page += 1

        print("[cinemapedia_quiz] Rebuilding artists and glossary…")
        sync_state["artistsStored"] = _rebuild_artists_and_glossary()
        data_version += 1

        print("\n[cinemapedia_quiz] ===== Sync Summary =====")
        print(f"  ✅ Films stored:   {sync_state['filmsStored']}")
        print(f"  ✅ Artists stored: {sync_state['artistsStored']}")
        print("[cinemapedia_quiz] ========================\n")
    except Exception as e:
        sync_state["error"] = str(e).split("\n[SQL:")[0][:500]
        print(f"\n[cinemapedia_quiz] ❌ Sync failed: {e}\n")
    finally:
        sync_state["running"] = False
        sync_state["finishedAt"] = _now()
        _sync_lock.release()


def clean_vector_store():
    global data_version
    try:
        vector_store.delete_collection()
        vector_store.create_collection()
        data_version += 1
        print("[cinemapedia_quiz] Vector store cleaned and collection recreated.")
        return {"status": "success"}, 200
    except Exception as e:
        return {"error": str(e)}, 500
