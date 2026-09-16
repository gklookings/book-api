"""
Cinemapedia quiz — question generation.

Every correct answer comes straight from the stored Cinemapedia data; the vector
store is used for random item selection (via cached metadata) and for picking
plausible "similar film" distractors. No LLM is called here.

Each question is built once with both English and Arabic (UAE MSA) text and then
rendered into the parallel `en` / `ar` arrays, so both arrays hold the same
questions in the same order with the same option ids.

Option types:
  - "text":    options are names / titles / years (`text` is set)
  - "picture": options are images (`picture` is set, `text` is null). `caption`
               holds the name and should only be shown after the user answers,
               otherwise it gives the answer away.
"""
import random
import re
import threading
import time
from collections import Counter

from app.langchain import cinemapedia_quiz as store

QUESTION_COUNTS = (8, 10, 12)
CATEGORIES = ("films", "actors", "directors", "mix")
LANGUAGES = ("en", "ar")
DEFAULT_LANGUAGE = "en"

OPTION_IDS = ("a", "b", "c", "d")
WRONG_OPTION_COUNT = len(OPTION_IDS) - 1
ARABIC_PRIORITY_RATIO = 0.8
CACHE_TTL_SECONDS = 1800
MAX_ATTEMPTS_PER_QUESTION = 30
DISTRACTOR_SAMPLE_SIZE = 400
MIN_DESCRIPTION_LENGTH = 60
CATEGORY_ROLES = {"actors": "actor", "directors": "director"}


# ── Cached pools ─────────────────────────────────────────────────────────────


class _SubjectPool:
    def __init__(self, items, key):
        self.items = items
        self.arabic = [i for i in items if i.get("has_arabic")]
        self.key = key

    def pick(self, rng, used, prefer_arabic):
        source = self.items
        if prefer_arabic and self.arabic and rng.random() < ARABIC_PRIORITY_RATIO:
            source = self.arabic
        if not source:
            return None
        for _ in range(50):
            item = rng.choice(source)
            if self.key(item) not in used:
                return item
        return None


def _film_key(film):
    return ("film", film["movie_id"])


def _artist_key(artist):
    return ("artist", artist["artist_id"])


class _Pools:
    def __init__(self, films, artists, glossary):
        self.films = films
        self.films_by_id = {f["movie_id"]: f for f in films}
        self.glossary = glossary
        self.countries = [c for c, ar in (glossary.get("countries") or {}).items() if ar]

        self.awards_by_festival = {}
        for film in films:
            for entry in film.get("nominations", []) + film.get("winners", []):
                self.awards_by_festival.setdefault(entry["festival"], set()).add(entry["award"])

        self.film_pools = {
            False: _SubjectPool(films, _film_key),
            True: _SubjectPool([f for f in films if f.get("has_picture")], _film_key),
        }
        self.artist_pools = {}
        for role in ("actor", "director"):
            with_role = [a for a in artists if role in a.get("roles", [])]
            self.artist_pools[role] = {
                False: _SubjectPool(with_role, _artist_key),
                True: _SubjectPool([a for a in with_role if a.get("has_picture")], _artist_key),
            }

    def subject_pool(self, category, needs_picture):
        if category == "films":
            return self.film_pools[needs_picture]
        return self.artist_pools[CATEGORY_ROLES[category]][needs_picture]

    def ar(self, kind, term):
        return (self.glossary.get(kind) or {}).get(term)


_cache_lock = threading.Lock()
_cache = {"pools": None, "loaded_at": 0.0, "version": None}


def _get_pools() -> _Pools:
    with _cache_lock:
        stale = (
            _cache["pools"] is None
            or _cache["version"] != store.data_version
            or time.time() - _cache["loaded_at"] > CACHE_TTL_SECONDS
        )
        if stale:
            glossary = next(iter(store.load_metadata("glossary")), {})
            _cache["pools"] = _Pools(
                store.load_metadata("film"), store.load_metadata("artist"), glossary
            )
            _cache["loaded_at"] = time.time()
            _cache["version"] = store.data_version
        return _cache["pools"]


# ── Small helpers ────────────────────────────────────────────────────────────


def _norm(value) -> str:
    return re.sub(r"\W+", "", (value or "").lower())


def _title(film, lang):
    return (film.get("title_ar") or film["title_en"]) if lang == "ar" else film["title_en"]


def _name(artist, lang):
    return (artist.get("name_ar") or artist["name_en"]) if lang == "ar" else artist["name_en"]


def _both(en, ar):
    return {"en": en, "ar": ar}


def _text_option(en, ar, is_right, picture=None):
    return {"text": _both(en, ar), "picture": picture, "caption": None, "isRight": is_right}


def _picture_option(picture, caption_en, caption_ar, is_right):
    return {"text": None, "picture": picture, "caption": _both(caption_en, caption_ar), "isRight": is_right}


def _film_text_option(film, is_right):
    return _text_option(_title(film, "en"), _title(film, "ar"), is_right)


def _film_picture_option(film, is_right):
    return _picture_option(film["picture"], _title(film, "en"), _title(film, "ar"), is_right)


def _artist_text_option(artist, is_right):
    return _text_option(_name(artist, "en"), _name(artist, "ar"), is_right)


def _artist_picture_option(artist, is_right):
    return _picture_option(artist["picture"], _name(artist, "en"), _name(artist, "ar"), is_right)


def _crew_ids(film):
    return {c["artistId"] for c in film.get("crew", [])}


def _role_words(artist, role):
    """(english noun, arabic noun, is_female)"""
    female = artist.get("gender") == "female"
    if role == "actor":
        return ("actress", "الممثلة", True) if female else ("actor", "الممثل", False)
    return ("director", "المخرجة", True) if female else ("director", "المخرج", False)


def _mask(text, names):
    for name in names:
        if name and len(name) >= 2:
            text = re.sub(re.escape(name), "…", text, flags=re.IGNORECASE)
    return text


class _Context:
    """Per-request state: pools, rng and distractor helpers."""

    def __init__(self, pools: _Pools, rng: random.Random):
        self.pools = pools
        self.rng = rng

    def _unique_adder(self, chosen, seen, limit, accept):
        def add(item, texts):
            if len(chosen) >= limit or not accept(item):
                return
            keys = [_norm(t) for t in texts]
            if any(k in s for k, s in zip(keys, seen)):
                return
            for k, s in zip(keys, seen):
                s.add(k)
            chosen.append(item)
        return add

    def film_distractors(self, film, accept=lambda f: True, need_picture=False):
        """Similar films first (vector search), then films from the same era."""
        chosen = []
        seen = ({_norm(_title(film, "en"))}, {_norm(_title(film, "ar"))})

        def ok(candidate):
            return (
                candidate["movie_id"] != film["movie_id"]
                and (not need_picture or candidate.get("has_picture"))
                and accept(candidate)
            )

        add = self._unique_adder(chosen, seen, WRONG_OPTION_COUNT, ok)
        query = f"{film['title_en']} {film.get('description_en', '')[:200]}"
        similar = [self.pools.films_by_id.get(i) for i in store.similar_film_ids(query)]
        similar = [f for f in similar if f]
        self.rng.shuffle(similar)
        for candidate in similar:
            add(candidate, (_title(candidate, "en"), _title(candidate, "ar")))

        if len(chosen) < WRONG_OPTION_COUNT:
            pool = self.pools.film_pools[need_picture].items
            sample = self.rng.sample(pool, min(DISTRACTOR_SAMPLE_SIZE, len(pool)))
            year = film.get("year") or 0
            sample.sort(key=lambda f: abs((f.get("year") or 0) - year) + self.rng.random() * 10)
            for candidate in sample:
                add(candidate, (_title(candidate, "en"), _title(candidate, "ar")))
        return chosen if len(chosen) == WRONG_OPTION_COUNT else None

    def artist_distractors(self, artist, role, accept=lambda a: True, need_picture=False):
        """Artists with the same role, preferring same gender, era and Arabic-ness."""
        chosen = []
        seen = ({_norm(_name(artist, "en"))}, {_norm(_name(artist, "ar"))})

        def ok(candidate):
            return candidate["artist_id"] != artist["artist_id"] and accept(candidate)

        add = self._unique_adder(chosen, seen, WRONG_OPTION_COUNT, ok)
        pool = self.pools.artist_pools[role][need_picture].items
        sample = self.rng.sample(pool, min(DISTRACTOR_SAMPLE_SIZE, len(pool)))
        year = artist.get("year") or 0
        gender = artist.get("gender")

        def score(candidate):
            value = abs((candidate.get("year") or 0) - year) + self.rng.random() * 8
            if gender and candidate.get("gender") and candidate["gender"] != gender:
                value += 1000
            if bool(candidate.get("has_arabic")) != bool(artist.get("has_arabic")):
                value += 15
            return value

        sample.sort(key=score)
        for candidate in sample:
            add(candidate, (_name(candidate, "en"), _name(candidate, "ar")))
        return chosen if len(chosen) == WRONG_OPTION_COUNT else None


def _question(q_en, q_ar, picture, option_type, right, wrongs, used_keys):
    return {
        "question": _both(q_en, q_ar),
        "picture": picture,
        "optionType": option_type,
        "options": [right] + wrongs,
        "usedKeys": used_keys,
    }


# ── Film question builders ───────────────────────────────────────────────────


def _film_identify_poster(ctx, film, role):
    wrongs = ctx.film_distractors(film)
    if not wrongs:
        return None
    return _question(
        "Which film is this poster from?",
        "إلى أي فيلم يعود هذا الملصق؟",
        film["picture"], "text",
        _film_text_option(film, True), [_film_text_option(f, False) for f in wrongs],
        [_film_key(film)],
    )


def _film_poster_pick(ctx, film, role):
    wrongs = ctx.film_distractors(film, need_picture=True)
    if not wrongs:
        return None
    return _question(
        f'Which of these is the poster of the film "{_title(film, "en")}"?',
        f"أيّ من هذه الملصقات يعود إلى فيلم «{_title(film, 'ar')}»؟",
        None, "picture",
        _film_picture_option(film, True), [_film_picture_option(f, False) for f in wrongs],
        [_film_key(film)] + [_film_key(f) for f in wrongs],
    )


def _film_year(ctx, film, role):
    year = film.get("year")
    if not year:
        return None
    max_year = max(year, time.gmtime().tm_year)
    offsets = [o for o in range(-12, 13) if o and 1890 <= year + o <= max_year]
    if len(offsets) < WRONG_OPTION_COUNT:
        return None
    wrong_years = [year + o for o in ctx.rng.sample(offsets, WRONG_OPTION_COUNT)]
    return _question(
        f'In which year was the film "{_title(film, "en")}" released?',
        f"في أي عام صدر فيلم «{_title(film, 'ar')}»؟",
        film["picture"], "text",
        _text_option(str(year), str(year), True),
        [_text_option(str(y), str(y), False) for y in wrong_years],
        [_film_key(film)],
    )


def _film_country(ctx, film, role):
    pools = ctx.pools
    countries = [c for c in film.get("countries", []) if pools.ar("countries", c)]
    candidates = [c for c in pools.countries if c not in film.get("countries", [])]
    if not countries or len(candidates) < WRONG_OPTION_COUNT:
        return None
    right = ctx.rng.choice(countries)
    wrongs = ctx.rng.sample(candidates, WRONG_OPTION_COUNT)
    return _question(
        f'Which of these countries took part in producing the film "{_title(film, "en")}"?',
        f"أيّ من هذه الدول شاركت في إنتاج فيلم «{_title(film, 'ar')}»؟",
        film["picture"], "text",
        _text_option(right, pools.ar("countries", right), True),
        [_text_option(c, pools.ar("countries", c), False) for c in wrongs],
        [_film_key(film)],
    )


def _film_award(ctx, film, role):
    pools = ctx.pools
    won = bool(film.get("winners"))
    entries = [
        e for e in (film.get("winners") or film.get("nominations") or [])
        if pools.ar("festivals", e["festival"]) and pools.ar("awards", e["award"])
    ]
    if not entries:
        return None
    entry = ctx.rng.choice(entries)
    festival = entry["festival"]
    film_awards = {
        e["award"] for e in film.get("winners", []) + film.get("nominations", [])
        if e["festival"] == festival
    }
    candidates = [
        a for a in pools.awards_by_festival.get(festival, set())
        if a not in film_awards and pools.ar("awards", a)
    ]
    if len(candidates) < WRONG_OPTION_COUNT:
        return None
    wrongs = ctx.rng.sample(sorted(candidates), WRONG_OPTION_COUNT)

    title_en, title_ar = _title(film, "en"), _title(film, "ar")
    festival_ar = pools.ar("festivals", festival)
    year_en = f" ({entry['year']})" if entry["year"] else ""
    if won:
        q_en = f'Which award did "{title_en}" win at the {festival}{year_en}?'
        q_ar = f"ما الجائزة التي فاز بها فيلم «{title_ar}» في {festival_ar}{year_en}؟"
    else:
        q_en = f'Which award was "{title_en}" nominated for at the {festival}{year_en}?'
        q_ar = f"ما الجائزة التي رُشّح لها فيلم «{title_ar}» في {festival_ar}{year_en}؟"
    return _question(
        q_en, q_ar, film["picture"], "text",
        _text_option(entry["award"], pools.ar("awards", entry["award"]), True),
        [_text_option(a, pools.ar("awards", a), False) for a in wrongs],
        [_film_key(film)],
    )


def _film_from_description(ctx, film, role):
    names = [film["title_en"], film.get("title_ar")]
    description_en = _mask(film.get("description_en") or "", names)
    description_ar = _mask(film.get("description_ar") or "", names) or description_en
    if len(description_en) < MIN_DESCRIPTION_LENGTH:
        return None
    wrongs = ctx.film_distractors(film, need_picture=True)
    if not wrongs:
        return None
    return _question(
        f'Which film matches this description? "{description_en}"',
        f"أيّ فيلم يتناول القصة التالية: «{description_ar}»؟",
        None, "picture",
        _film_picture_option(film, True), [_film_picture_option(f, False) for f in wrongs],
        [_film_key(film)] + [_film_key(f) for f in wrongs],
    )


# ── Artist question builders (actors + directors) ────────────────────────────


def _artist_identify(ctx, artist, role):
    wrongs = ctx.artist_distractors(artist, role)
    if not wrongs:
        return None
    noun_en, noun_ar, female = _role_words(artist, role)
    return _question(
        f"Who is this {noun_en}?",
        f"مَن هذه {noun_ar}؟" if female else f"مَن هذا {noun_ar}؟",
        artist["picture"], "text",
        _artist_text_option(artist, True), [_artist_text_option(a, False) for a in wrongs],
        [_artist_key(artist)],
    )


def _artist_pick_picture(ctx, artist, role):
    wrongs = ctx.artist_distractors(artist, role, need_picture=True)
    if not wrongs:
        return None
    noun_en, noun_ar, _ = _role_words(artist, role)
    return _question(
        f"Which of these photos shows the {noun_en} {_name(artist, 'en')}?",
        f"أيّ من هذه الصور تعود إلى {noun_ar} {_name(artist, 'ar')}؟",
        None, "picture",
        _artist_picture_option(artist, True), [_artist_picture_option(a, False) for a in wrongs],
        [_artist_key(artist)] + [_artist_key(a) for a in wrongs],
    )


def _artist_film(ctx, artist, role):
    film_ids = artist.get("acted_in" if role == "actor" else "directed") or []
    films = [ctx.pools.films_by_id[i] for i in film_ids if i in ctx.pools.films_by_id]
    if not films:
        return None
    film = ctx.rng.choice(films)
    wrongs = ctx.film_distractors(
        film, accept=lambda f: artist["artist_id"] not in _crew_ids(f)
    )
    if not wrongs:
        return None
    noun_en, noun_ar, female = _role_words(artist, role)
    name_ar = _name(artist, "ar")
    if role == "actor":
        q_en = f"In which of these films did this {noun_en}, {_name(artist, 'en')}, appear?"
        q_ar = f"في أيّ من هذه الأفلام {'شاركت' if female else 'شارك'} {noun_ar} {name_ar}؟"
    else:
        q_en = f"Which of these films was directed by {_name(artist, 'en')}?"
        q_ar = f"أيّ من هذه الأفلام من إخراج {name_ar}؟"
    return _question(
        q_en, q_ar, artist["picture"], "text",
        _film_text_option(film, True), [_film_text_option(f, False) for f in wrongs],
        [_artist_key(artist), _film_key(film)],
    )


def _film_artist(ctx, artist, role):
    film_ids = artist.get("acted_in" if role == "actor" else "directed") or []
    films = [
        ctx.pools.films_by_id[i] for i in film_ids
        if i in ctx.pools.films_by_id and ctx.pools.films_by_id[i].get("has_picture")
    ]
    if not films:
        return None
    film = ctx.rng.choice(films)
    crew = _crew_ids(film)
    wrongs = ctx.artist_distractors(
        artist, role, accept=lambda a: a["artist_id"] not in crew
    )
    if not wrongs:
        return None
    title_en, title_ar = _title(film, "en"), _title(film, "ar")
    if role == "actor":
        q_en = f'Which of these actors appears in the film "{title_en}"?'
        q_ar = f"مَن من هؤلاء شارك في تمثيل فيلم «{title_ar}»؟"
    else:
        q_en = f'Who directed the film "{title_en}"?'
        q_ar = f"مَن أخرج فيلم «{title_ar}»؟"
    return _question(
        q_en, q_ar, film["picture"], "text",
        _artist_text_option(artist, True), [_artist_text_option(a, False) for a in wrongs],
        [_artist_key(artist), _film_key(film)],
    )


# type -> (builder, subject needs a valid picture, weight)
QUESTION_TYPES = {
    "films": {
        "film_identify_poster": (_film_identify_poster, True, 4),
        "film_poster_pick": (_film_poster_pick, True, 2),
        "film_year": (_film_year, True, 2),
        "film_country": (_film_country, True, 2),
        "film_award": (_film_award, True, 2),
        "film_from_description": (_film_from_description, True, 1),
    },
    "people": {
        "artist_identify": (_artist_identify, True, 4),
        "artist_pick_picture": (_artist_pick_picture, True, 3),
        "artist_film": (_artist_film, True, 2),
        "film_artist": (_film_artist, False, 2),
    },
}


def _types_for(category):
    return QUESTION_TYPES["films" if category == "films" else "people"]


def _choose_type(category, usage, rng):
    types = _types_for(category)
    names = list(types)
    weights = [types[n][2] / (1 + 2 * usage[(category, n)]) for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _slot_categories(category, qn_count, rng):
    if category != "mix":
        return [category] * qn_count
    order = ["films", "actors", "directors"]
    rng.shuffle(order)
    slots = [order[i % len(order)] for i in range(qn_count)]
    rng.shuffle(slots)
    return slots


def _render(question, index, lang):
    return {
        "id": index + 1,
        "category": question["category"],
        "type": question["type"],
        "question": question["question"][lang],
        "picture": question["picture"],
        "isPicAvailable": bool(question["picture"]),
        "optionType": question["optionType"],
        "options": [
            {
                "id": OPTION_IDS[i],
                "text": option["text"][lang] if option["text"] else None,
                "picture": option["picture"],
                "isPicAvailable": bool(option["picture"]),
                "caption": option["caption"][lang] if option["caption"] else None,
                "isRight": option["isRight"],
            }
            for i, option in enumerate(question["options"])
        ],
    }


# ── Public entry point ───────────────────────────────────────────────────────


def generate_quiz(qn_count: int, category: str, language: str):
    try:
        pools = _get_pools()
        if not pools.films:
            return {
                "error": "Cinemapedia quiz data is empty. Run POST /cinemapedia-quiz/sync first."
            }, 404

        rng = random.Random()
        ctx = _Context(pools, rng)
        prefer_arabic = language == "ar"
        used = set()
        usage = Counter()
        questions = []

        for slot_category in _slot_categories(category, qn_count, rng):
            role = CATEGORY_ROLES.get(slot_category)
            built = None
            for _ in range(MAX_ATTEMPTS_PER_QUESTION):
                qtype = _choose_type(slot_category, usage, rng)
                builder, needs_picture, _ = _types_for(slot_category)[qtype]
                subject = pools.subject_pool(slot_category, needs_picture).pick(
                    rng, used, prefer_arabic
                )
                if subject is None:
                    continue
                built = builder(ctx, subject, role)
                if built and not any(k in used for k in built["usedKeys"]):
                    built.update(category=slot_category, type=qtype)
                    break
                built = None
            if not built:
                return {
                    "error": f"Not enough data to build {qn_count} '{category}' questions."
                }, 500
            used.update(built["usedKeys"])
            usage[(slot_category, built["type"])] += 1
            rng.shuffle(built["options"])
            questions.append(built)

        return {
            "qnCount": qn_count,
            "category": category,
            "language": language,
            "en": [_render(q, i, "en") for i, q in enumerate(questions)],
            "ar": [_render(q, i, "ar") for i, q in enumerate(questions)],
        }, 200
    except Exception as e:
        print(f"[cinemapedia_quiz] Error generating quiz: {e}")
        return {"error": str(e)}, 500
