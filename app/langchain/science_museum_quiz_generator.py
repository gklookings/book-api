"""
Science Museum quiz — question generation.

Every correct answer comes straight from the Science Museum API data fetched
for this request (see science_museum_quiz). The LLM is only used to translate
the names / description snippets a quiz uses into the other language; the
question wording itself is written here in English and Arabic (UAE MSA).

Questions are built in the requested language and rendered into parallel
`en` / `ar` arrays (same questions, same order, same option ids), like the
Cinemapedia quiz. Nobel citations only exist in English, so the Arabic
questions quote them in English.

`coverImage` is the person's portrait; it is the question picture for
"who is this" style questions and the option picture for "pick the photo" ones.

Option types:
  - "text":    options are names / years / categories (`text` is set)
  - "picture": options are portraits (`picture` is set, `text` is null).
               `caption` holds the name and should only be shown after the
               user answers, otherwise it gives the answer away.

The API's isNobel flag has false negatives (e.g. Marie Curie is not marked),
so no question ever relies on someone *not* having won a Nobel Prize.
"""
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app.langchain import science_museum_quiz as api

MIN_QUESTION_COUNT = 1
MAX_QUESTION_COUNT = 99
LANGUAGES = ("en", "ar")
DEFAULT_LANGUAGE = "en"

OPTION_IDS = ("a", "b", "c", "d")
WRONG_OPTION_COUNT = len(OPTION_IDS) - 1
MAX_ATTEMPTS_PER_QUESTION = 30
# Scientists fetched per request: enough subjects + distractors without a slow call.
SUBJECTS_PER_QUESTION = 4
MIN_POOL_SIZE = 40
MAX_POOL_SIZE = 120
# With a fieldId, small fields (some have 1–4 people) borrow distractors from this pool.
DISTRACTOR_POOL_SIZE = 40
MIN_DESCRIPTION_LENGTH = 80
DESCRIPTION_QUESTION_LIMIT = 300
FIRST_NOBEL_YEAR = 1901

# key -> (category en, category ar, prize en, prize ar)
NOBEL_CATEGORIES = {
    "physics": ("Physics", "الفيزياء", "Nobel Prize in Physics", "جائزة نوبل في الفيزياء"),
    "chemistry": ("Chemistry", "الكيمياء", "Nobel Prize in Chemistry", "جائزة نوبل في الكيمياء"),
    "medicine": (
        "Physiology or Medicine", "الطب",
        "Nobel Prize in Physiology or Medicine", "جائزة نوبل في الطب",
    ),
    "literature": ("Literature", "الأدب", "Nobel Prize in Literature", "جائزة نوبل في الأدب"),
    "peace": ("Peace", "السلام", "Nobel Peace Prize", "جائزة نوبل للسلام"),
    "economics": (
        "Economic Sciences", "العلوم الاقتصادية",
        "Nobel Prize in Economic Sciences", "جائزة نوبل في العلوم الاقتصادية",
    ),
}

# Museum field a Nobel category belongs to (ids from json_fields.php).
NOBEL_FIELD_IDS = {
    "physics": 1, "chemistry": 2, "medicine": 6,
    "literature": 12, "peace": 13, "economics": 14,
}
# Nobel categories whose laureates might also belong to a museum field (e.g. a
# Physics laureate can be listed under Astronomy). Laureates of these categories
# are never used as "not in this field" distractors.
FIELD_CONFLICTS = {
    1: {"physics", "chemistry"},
    2: {"chemistry", "physics", "medicine"},
    3: {"medicine", "chemistry"},
    4: {"physics"},
    5: {"physics", "economics"},
    6: {"medicine", "chemistry"},
    7: {"physics", "chemistry"},
    9: {"medicine", "economics"},
    10: {"physics", "chemistry"},
    11: {"literature", "peace"},
    12: {"literature", "peace"},
    13: {"peace", "literature", "economics"},
    14: {"economics", "peace"},
}
DEFAULT_FIELD_CONFLICTS = {"physics", "chemistry", "medicine"}

AR_CENTURIES = (
    None, "الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع",
    "الثامن", "التاسع", "العاشر", "الحادي عشر", "الثاني عشر", "الثالث عشر",
    "الرابع عشر", "الخامس عشر", "السادس عشر", "السابع عشر", "الثامن عشر",
    "التاسع عشر", "العشرون", "الحادي والعشرون",
)


# ── Bilingual values ─────────────────────────────────────────────────────────


class _Translatable:
    """Text in the request language that has to be translated for the other one."""

    __slots__ = ("kind", "text")

    def __init__(self, kind, text):
        self.kind = kind
        self.text = text

    @property
    def key(self):
        return (self.kind, self.text)


def _both(en, ar):
    return {"en": en, "ar": ar}


def _fmt(en, ar, **params):
    return {"en": en, "ar": ar, "params": params}


def _collect(value, out):
    if isinstance(value, _Translatable):
        out.add(value.key)
    elif isinstance(value, dict):
        for inner in value.values():
            _collect(inner, out)


def _resolve(value, lang, source_lang, translations):
    if value is None:
        return None
    if isinstance(value, _Translatable):
        if lang == source_lang:
            return value.text
        return translations.get(value.key) or value.text
    if isinstance(value, dict):
        if "params" in value:
            params = {
                k: _resolve(v, lang, source_lang, translations) for k, v in value["params"].items()
            }
            return value[lang].format(**params)
        return _resolve(value[lang], lang, source_lang, translations)
    return str(value)


# ── Small helpers ────────────────────────────────────────────────────────────


def _norm(value) -> str:
    return re.sub(r"\W+", "", (value or "").lower())


def _key(scientist):
    return scientist["id"]


def _name(scientist):
    return _Translatable("name", scientist["name"])


def _text_option(text, is_right):
    return {"text": text, "picture": None, "caption": None, "isRight": is_right}


def _name_option(scientist, is_right):
    return _text_option(_name(scientist), is_right)


def _picture_option(scientist, is_right):
    return {"text": None, "picture": scientist["picture"], "caption": _name(scientist), "isRight": is_right}


def _category(key, prize=False):
    cat_en, cat_ar, prize_en, prize_ar = NOBEL_CATEGORIES[key]
    return _both(prize_en, prize_ar) if prize else _both(cat_en, cat_ar)


def _categories(scientist):
    return {n["category"] for n in scientist["nobels"]}


def _ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _century_option(century, is_right):
    return _text_option(
        _both(f"{_ordinal(century)} century", f"القرن {AR_CENTURIES[century]} الميلادي"), is_right
    )


def _mask(text, name):
    """Hide the person's name (and each part of it) in a description."""
    parts = [name] + [p for p in re.split(r"[\s.,\-]+", name) if len(p) >= 3]
    for part in sorted(set(parts), key=len, reverse=True):
        # Optional Arabic prefix letter (و ف ب ل ك) so "لرامزي" is masked too.
        text = re.sub(rf"(?<!\w)([وفبلك]?){re.escape(part)}(?!\w)", r"\1…", text, flags=re.I)
    return re.sub(r"…(\s*…)+", "…", text)


def _question(q, picture, option_type, right, wrongs, subject):
    return {
        "question": q,
        "picture": picture,
        "optionType": option_type,
        "options": [right] + wrongs,
        "subject": subject,
    }


class _Context:
    """Per-request state: scientist pools, fields, rng and distractor helpers."""

    def __init__(self, subjects, extra, fields, rng, field_id):
        self.subjects = subjects
        self.subject_ids = {_key(s) for s in subjects}
        self.pool = subjects + [s for s in extra if _key(s) not in self.subject_ids]
        self.laureates = [s for s in self.pool if s["nobels"]]
        self.fields = fields
        self.rng = rng
        self.field_id = field_id

    def field_of(self, scientist):
        if scientist.get("fieldId"):
            return scientist["fieldId"]
        field_ids = {NOBEL_FIELD_IDS[c] for c in _categories(scientist)}
        return field_ids.pop() if len(field_ids) == 1 else None

    def distractors(self, subject, candidates=None, accept=lambda s: True, score=None):
        """Three people with distinct names; lowest score first (plus a little noise)."""
        year = subject["birthYear"]

        def default_score(candidate):
            value = abs(candidate["birthYear"] - year) / 10
            # With a fieldId, prefer people from the same field.
            if self.field_id and _key(candidate) not in self.subject_ids:
                value += 50
            return value

        score = score or default_score
        candidates = [
            c for c in (self.pool if candidates is None else candidates)
            if _key(c) != _key(subject) and accept(c)
        ]
        candidates.sort(key=lambda c: score(c) + self.rng.random() * 5)
        chosen, seen = [], {_norm(subject["name"])}
        for candidate in candidates:
            name = _norm(candidate["name"])
            if name in seen:
                continue
            seen.add(name)
            chosen.append(candidate)
            if len(chosen) == WRONG_OPTION_COUNT:
                return chosen
        return None


# ── Scientist question builders ──────────────────────────────────────────────


def _scientist_identify(ctx, s):
    wrongs = ctx.distractors(s)
    if not wrongs:
        return None
    return _question(
        _both("Who is the person in this picture?", "مَن الشخصية التي تظهر في هذه الصورة؟"),
        s["picture"], "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


def _scientist_pick_picture(ctx, s):
    wrongs = ctx.distractors(s)
    if not wrongs:
        return None
    return _question(
        _fmt("Which of these pictures shows {name}?", "أيّ من هذه الصور تعود إلى {name}؟", name=_name(s)),
        None, "picture",
        _picture_option(s, True), [_picture_option(w, False) for w in wrongs], s,
    )


def _scientist_from_description(ctx, s):
    description = api.shorten(_mask(s["description"], s["name"]), DESCRIPTION_QUESTION_LIMIT)
    if len(description) < MIN_DESCRIPTION_LENGTH:
        return None
    wrongs = ctx.distractors(s)
    if not wrongs:
        return None
    return _question(
        _fmt(
            'Who is described here? "{description}"',
            "عمّن يتحدث هذا الوصف: «{description}»؟",
            description=_Translatable("text", description),
        ),
        None, "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


# ── Timeline question builders ───────────────────────────────────────────────


def _birth_year(ctx, s):
    year = s["birthYear"]
    max_year = time.gmtime().tm_year
    offsets = [o for o in range(-15, 16) if o and year + o <= max_year]
    wrong_years = [year + o for o in ctx.rng.sample(offsets, WRONG_OPTION_COUNT)]
    return _question(
        _fmt("In which year was {name} born?", "ما سنة ميلاد {name}؟", name=_name(s)),
        s["picture"], "text",
        _text_option(str(year), True), [_text_option(str(y), False) for y in wrong_years], s,
    )


def _birth_century(ctx, s):
    century = (s["birthYear"] - 1) // 100 + 1
    candidates = [c for c in range(century - 3, century + 4) if c != century and 1 <= c < len(AR_CENTURIES)]
    if len(candidates) < WRONG_OPTION_COUNT:
        return None
    wrongs = ctx.rng.sample(candidates, WRONG_OPTION_COUNT)
    return _question(
        _fmt("In which century was {name} born?", "إلى أي قرن يعود ميلاد {name}؟", name=_name(s)),
        s["picture"], "text",
        _century_option(century, True), [_century_option(c, False) for c in wrongs], s,
    )


def _born_first(ctx, s):
    year = s["birthYear"]
    # Everyone else is born at least 5 years later, so the answer is unambiguous.
    wrongs = ctx.distractors(
        s,
        accept=lambda c: 5 <= c["birthYear"] - year <= 150,
        score=lambda c: (c["birthYear"] - year) / 10,
    )
    if not wrongs:
        return None
    return _question(
        _both("Which of these people was born first?", "أيّ من هؤلاء وُلد أولاً؟"),
        None, "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


# ── Field question builders ──────────────────────────────────────────────────


def _field_pick(ctx, s):
    field_id = ctx.field_of(s)
    field = ctx.fields.get(field_id)
    if not field:
        return None
    conflicts = FIELD_CONFLICTS.get(field_id, DEFAULT_FIELD_CONFLICTS)
    # Only laureates have a known field, so only they can safely be "not in this field".
    wrongs = ctx.distractors(
        s,
        candidates=ctx.laureates,
        accept=lambda c: not (_categories(c) & conflicts) and not c.get("fieldId"),
    )
    if not wrongs:
        return None
    return _question(
        _fmt(
            "Which of these people is associated with the field of {field}?",
            "أيّ من هؤلاء يرتبط اسمه بمجال {field}؟",
            field=_both(field["en"], field["ar"]),
        ),
        field.get("picture"), "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


# ── Nobel question builders ──────────────────────────────────────────────────


def _nobel_category(ctx, s):
    won = _categories(s)
    candidates = [c for c in NOBEL_CATEGORIES if c not in won]
    if len(candidates) < WRONG_OPTION_COUNT:
        return None
    wrongs = ctx.rng.sample(candidates, WRONG_OPTION_COUNT)
    return _question(
        _fmt(
            "In which category was {name} awarded the Nobel Prize?",
            "ما فئة جائزة نوبل التي مُنحت إلى {name}؟",
            name=_name(s),
        ),
        s["picture"], "text",
        _text_option(_category(ctx.rng.choice(sorted(won))), True),
        [_text_option(_category(c), False) for c in wrongs], s,
    )


def _nobel_year(ctx, s):
    entry = ctx.rng.choice(s["nobels"])
    won_years = {n["year"] for n in s["nobels"]}
    max_year = time.gmtime().tm_year
    years = [
        entry["year"] + o for o in range(-12, 13)
        if FIRST_NOBEL_YEAR <= entry["year"] + o <= max_year and entry["year"] + o not in won_years
    ]
    if len(years) < WRONG_OPTION_COUNT:
        return None
    wrong_years = ctx.rng.sample(years, WRONG_OPTION_COUNT)
    return _question(
        _fmt(
            "In which year was {name} awarded the {prize}?",
            "في أي عام مُنحت {prize} إلى {name}؟",
            name=_name(s), prize=_category(entry["category"], prize=True),
        ),
        s["picture"], "text",
        _text_option(str(entry["year"]), True),
        [_text_option(str(y), False) for y in wrong_years], s,
    )


def _laureate_distractors(ctx, s, entry, accept):
    """Other laureates, preferring the same category and nearby prize years."""
    return ctx.distractors(
        s,
        candidates=ctx.laureates,
        accept=accept,
        score=lambda c: (0 if entry["category"] in _categories(c) else 50)
        + min(abs(n["year"] - entry["year"]) for n in c["nobels"]) / 5,
    )


def _not_co_laureate(entry):
    prize = (entry["year"], entry["category"])
    return lambda c: all((n["year"], n["category"]) != prize for n in c["nobels"])


def _nobel_identify(ctx, s):
    entry = ctx.rng.choice(s["nobels"])
    wrongs = _laureate_distractors(ctx, s, entry, _not_co_laureate(entry))
    if not wrongs:
        return None
    return _question(
        _fmt(
            "This person was awarded the {prize} in {year}. Who is it?",
            "مَن الشخصية التي تظهر في الصورة، والتي مُنحت {prize} عام {year}؟",
            prize=_category(entry["category"], prize=True), year=entry["year"],
        ),
        s["picture"], "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


def _nobel_category_pick(ctx, s):
    entry = ctx.rng.choice(s["nobels"])
    wrongs = _laureate_distractors(
        ctx, s, entry, lambda c: entry["category"] not in _categories(c)
    )
    if not wrongs:
        return None
    return _question(
        _fmt(
            "Which of these people was awarded the {prize}?",
            "أيّ من هؤلاء حصل على {prize}؟",
            prize=_category(entry["category"], prize=True),
        ),
        None, "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


def _nobel_citation(ctx, s):
    entries = [n for n in s["nobels"] if len(n["citation"]) >= 15]
    if not entries:
        return None
    entry = ctx.rng.choice(entries)
    citation = _mask(entry["citation"], s["name"])
    citation_key = _norm(entry["citation"])
    # Co-laureates often share the exact citation.
    wrongs = _laureate_distractors(
        ctx, s, entry,
        lambda c: all(_norm(n["citation"]) != citation_key for n in c["nobels"]),
    )
    if not wrongs:
        return None
    return _question(
        # Citations only exist in English, so the Arabic question quotes them as is.
        _fmt(
            'Who was awarded the {prize} "{citation}"?',
            "لمن مُنحت {prize} «{citation}»؟",
            prize=_category(entry["category"], prize=True), citation=citation,
        ),
        None, "text",
        _name_option(s, True), [_name_option(w, False) for w in wrongs], s,
    )


def _nobel_winner_of_year(ctx, s):
    entry = ctx.rng.choice(s["nobels"])
    wrongs = _laureate_distractors(ctx, s, entry, _not_co_laureate(entry))
    if not wrongs:
        return None
    return _question(
        _fmt(
            "Who was awarded the {prize} in {year}?",
            "لمن مُنحت {prize} عام {year}؟",
            prize=_category(entry["category"], prize=True), year=entry["year"],
        ),
        None, "picture",
        _picture_option(s, True), [_picture_option(w, False) for w in wrongs], s,
    )


def _always(ctx, s):
    return True


def _is_laureate(ctx, s):
    return bool(s["nobels"])


# type -> (builder, eligible(ctx, subject), weight, category)
QUESTION_TYPES = {
    "scientist_identify": (_scientist_identify, _always, 4, "scientist"),
    "scientist_pick_picture": (_scientist_pick_picture, _always, 3, "scientist"),
    "scientist_from_description": (
        _scientist_from_description,
        lambda ctx, s: len(s["description"]) >= MIN_DESCRIPTION_LENGTH, 1, "scientist",
    ),
    "birth_year": (_birth_year, lambda ctx, s: s["birthYear"] >= 1750, 2, "timeline"),
    "birth_century": (_birth_century, lambda ctx, s: s["birthYear"] < 1900, 1, "timeline"),
    "born_first": (_born_first, _always, 1, "timeline"),
    "field_pick": (_field_pick, lambda ctx, s: ctx.field_of(s) in ctx.fields, 2, "field"),
    # Within a fieldId quiz the category is (nearly) always that field, so it gives itself away.
    "nobel_category": (_nobel_category, lambda ctx, s: s["nobels"] and not ctx.field_id, 2, "nobel"),
    "nobel_year": (_nobel_year, _is_laureate, 2, "nobel"),
    "nobel_citation": (_nobel_citation, _is_laureate, 2, "nobel"),
    "nobel_winner_of_year": (_nobel_winner_of_year, _is_laureate, 2, "nobel"),
    "nobel_identify": (_nobel_identify, _is_laureate, 2, "nobel"),
    "nobel_category_pick": (_nobel_category_pick, _is_laureate, 1, "nobel"),
}
NOBEL_FIELD = {"en": "Nobel Prize", "ar": "جائزة نوبل"}


def _choose_type(eligible, usage, rng):
    names = [n for n in eligible if eligible[n]]
    weights = [QUESTION_TYPES[n][2] / (1 + 2 * usage[n]) for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _pick_subject(candidates, qtype, used, subject_usage, rng):
    """A subject not yet asked with this type, preferring the least used ones."""
    fresh = [s for s in candidates if (_key(s), qtype) not in used]
    if not fresh:
        return None
    least = min(subject_usage[_key(s)] for s in fresh)
    return rng.choice([s for s in fresh if subject_usage[_key(s)] == least])


def _render(question, index, lang, source_lang, translations):
    def value(v):
        return _resolve(v, lang, source_lang, translations)

    return {
        "id": index + 1,
        "category": question["category"],
        "type": question["type"],
        "question": value(question["question"]),
        "picture": question["picture"],
        "isPicAvailable": bool(question["picture"]),
        "optionType": question["optionType"],
        "options": [
            {
                "id": OPTION_IDS[i],
                "text": value(option["text"]),
                "picture": option["picture"],
                "isPicAvailable": bool(option["picture"]),
                "caption": value(option["caption"]),
                "isRight": option["isRight"],
            }
            for i, option in enumerate(question["options"])
        ],
    }


# ── Public entry point ───────────────────────────────────────────────────────


def generate_quiz(qn_count: int, language: str, field_id: int | None = None, is_nobel: bool = False):
    """is_nobel: only Nobel laureates and only "nobel" category questions."""
    try:
        pool_size = min(MAX_POOL_SIZE, max(MIN_POOL_SIZE, qn_count * SUBJECTS_PER_QUESTION))
        with ThreadPoolExecutor(max_workers=3) as pool:
            subjects_job = pool.submit(api.fetch_scientists, language, pool_size, field_id, is_nobel)
            extra_job = (
                pool.submit(api.fetch_scientists, language, DISTRACTOR_POOL_SIZE, None, is_nobel)
                if field_id else None
            )
            fields_job = pool.submit(api.get_fields)
            subjects = subjects_job.result()
            extra = extra_job.result() if extra_job else []
            fields = fields_job.result()

        filters = " and ".join(
            f for f in (f"fieldId {field_id}" if field_id else "", "isNobel" if is_nobel else "") if f
        )
        suffix = f" for {filters}" if filters else ""
        if not subjects:
            return {"error": f"No scientists found{suffix}."}, 404

        rng = random.Random()
        ctx = _Context(subjects, extra, fields, rng, field_id)
        eligible = {
            name: [s for s in subjects if check(ctx, s)]
            for name, (_, check, _, category) in QUESTION_TYPES.items()
            if not is_nobel or category == "nobel"
        }
        used = set()
        usage = Counter()
        subject_usage = Counter()
        questions = []

        for _ in range(qn_count):
            built = None
            for _ in range(MAX_ATTEMPTS_PER_QUESTION):
                if not any(eligible.values()):
                    break
                qtype = _choose_type(eligible, usage, rng)
                subject = _pick_subject(eligible[qtype], qtype, used, subject_usage, rng)
                if subject is None:
                    eligible[qtype] = []
                    continue
                builder, _, _, category = QUESTION_TYPES[qtype]
                built = builder(ctx, subject)
                used.add((_key(subject), qtype))
                if built:
                    built.update(category=category, type=qtype)
                    break
            if not built:
                return {"error": f"Not enough data to build {qn_count} questions{suffix}."}, 500
            usage[built["type"]] += 1
            subject_usage[_key(built["subject"])] += 1
            rng.shuffle(built["options"])
            questions.append(built)

        pending = set()
        for question in questions:
            _collect(question["question"], pending)
            for option in question["options"]:
                _collect(option["text"], pending)
                _collect(option["caption"], pending)
        translations = api.translate(pending, language)

        field = fields.get(field_id) if field_id else None
        if field:
            field = _both(field["en"], field["ar"])
        elif is_nobel:
            # The frontend treats Nobel as a field of its own.
            field = NOBEL_FIELD
        return {
            "qnCount": qn_count,
            "fieldId": field_id,
            "isNobel": is_nobel,
            "field": field,
            "language": language,
            "en": [_render(q, i, "en", language, translations) for i, q in enumerate(questions)],
            "ar": [_render(q, i, "ar", language, translations) for i, q in enumerate(questions)],
        }, 200
    except api.ScienceMuseumApiError as e:
        print(f"[science_museum_quiz] {e}")
        return {"error": str(e)}, 502
    except Exception as e:
        print(f"[science_museum_quiz] Error generating quiz: {e}")
        return {"error": str(e)}, 500
