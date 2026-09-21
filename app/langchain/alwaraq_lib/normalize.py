"""
Arabic text normalization for Ask Alwaraq.

Used for keyword search and quote verification only.
The original text is never modified or stored in normalized form in `books`.
"""

import re

# Tashkeel (harakat, tanween, shadda, sukun), superscript alef and
# Quranic annotation marks.
TASHKEEL_CHARS = (
    "ؘؙؚؐؑؒؓؔؕؖؗ"
    "ًٌٍَُِّْٕٓٔ"
    "ٰٖٜٟٗ٘ٙٚٛٝٞ"
    "ۖۗۘۙۚۛۜ۟۠ۡۢ"
    "ۣ۪ۭۤۧۨ۫۬"
)
TATWEEL = "ـ"

# Characters removed before comparison (tashkeel + tatweel).
REMOVED_CHARS = TASHKEEL_CHARS + TATWEEL

# Letter-form folding: source chars -> target chars (same length, used by
# both str.translate here and SQL translate() in retrieval.py).
FOLD_FROM = "أإآٱىةؤئ"
FOLD_TO = "اااايهوي"

_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

_TRANSLATION = str.maketrans(
    {
        **{c: None for c in REMOVED_CHARS},
        **{f: t for f, t in zip(FOLD_FROM, FOLD_TO)},
        **{d: str(i) for i, d in enumerate(_ARABIC_INDIC_DIGITS)},
        **{d: str(i) for i, d in enumerate(_PERSIAN_DIGITS)},
    }
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ARABIC_LETTER_RE = re.compile(r"[ء-ي]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def normalize_arabic(text: str) -> str:
    """Strip diacritics/tatweel, fold letter variants, unify digits and spaces."""
    if not text:
        return ""
    text = text.translate(_TRANSLATION)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """normalize_arabic + punctuation removal + lowercase; for quote matching."""
    text = normalize_arabic(text)
    text = _PUNCT_RE.sub(" ", text).replace("_", " ")
    return _SPACE_RE.sub(" ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return normalize_for_match(text).split()


def detect_language(text: str) -> str:
    """'ar' when Arabic letters dominate, else 'en'."""
    if not text:
        return "ar"
    ar = len(_ARABIC_LETTER_RE.findall(text))
    en = len(_LATIN_LETTER_RE.findall(text))
    return "ar" if ar >= en else "en"


# ── Spelling variants (keyword search) ───────────────────────────────────────
#
# Folding letter forms is not enough for foreign names: the library's older
# translations write "أميركة" / "فرنسة" where a reader's question writes
# "أمريكا" / "فرنسا". `translate()` cannot fix that — the letters differ in
# order, not in form — so the query side expands each keyword into the handful
# of spellings the corpus actually uses, and the search matches any of them.

# Applied in both directions, on already-normalized text.
_SWAP_RULES = (
    ("امريك", "اميرك"),      # أمريكا / أميركا
    ("انكلتر", "انجلتر"),    # إنكلترة / إنجلترا
    ("اوربا", "اوروبا"),     # أوربا / أوروبا
)
MAX_VARIANTS = 4


def _swap_variants(term: str) -> set[str]:
    out = {term}
    for a, b in _SWAP_RULES:
        for src, dst in ((a, b), (b, a)):
            out |= {t.replace(src, dst) for t in out if src in t}
    return out


def _final_alef_variants(term: str) -> set[str]:
    """
    أميركا / أميركة — a name ending in ا and the same name ending in ة (folded
    to ه) are one name. Only the term's last word is varied: that is where the
    foreign name sits ("ثلاث مقالات عن أمريكا"), and varying every word would
    multiply out into patterns that cost search time without adding recall.
    """
    head, _, last = term.rpartition(" ")
    out = {term}
    for ending, other in (("ا", "ه"), ("ه", "ا")):
        if last.endswith(ending):
            swapped = last[:-1] + other
            out.add(f"{head} {swapped}".strip() if head else swapped)
    return out


def spelling_variants(term: str, max_variants: int = MAX_VARIANTS) -> list[str]:
    """
    Spellings of `term` (already normalized) that the corpus may use instead.
    The term itself always comes first; at most `max_variants` are returned.
    """
    term = normalize_arabic(term)
    if not term or not _ARABIC_LETTER_RE.search(term):
        return [term] if term else []
    variants = set()
    for swapped in _swap_variants(term):
        variants |= _final_alef_variants(swapped)
    variants.discard(term)
    return [term] + sorted(variants)[: max(0, max_variants - 1)]


# ── Literal terms lifted straight out of the question ────────────────────────

# A quoted span, in any script and any of the quote marks readers actually type.
_QUOTED_RE = re.compile(r"[\"«“”']([^\"«»“”']{2,80})[\"»“”']")
_CONNECTORS = r"of|on|in|at|and|or|the|a|an|for|to|de|del|van|von|du|le|la|el|al|bin|ibn"
# A Title Case run of two or more words: an English work or name as typed,
# e.g. "Three Essays On America", "The Ordeal of Mark Twain", "Van Wyck Brooks".
_TITLE_RUN_RE = re.compile(
    rf"\b[A-Z][\w'’\-]+(?:\s+(?:{_CONNECTORS})\b|\s+[A-Z][\w'’\-]+)+", re.UNICODE
)
_TRAILING_CONNECTOR_RE = re.compile(rf"\s+(?:{_CONNECTORS})$", re.IGNORECASE)


def literal_terms(text: str, max_terms: int = 3) -> list[str]:
    """
    Distinctive phrases to search for exactly as the reader wrote them.

    The rewrite/translation steps are what usually lose a title: a question
    about "Three Essays On America" comes back as "ثلاث مقالات عن أمريكا", and
    the English title — often the one string that appears verbatim in the text
    — is never searched. These terms keep it in the query.
    """
    found: list[str] = []
    for match in _QUOTED_RE.findall(text or ""):
        phrase = " ".join(match.split())
        if len(phrase) > 2:
            found.append(phrase)
    for match in _TITLE_RUN_RE.findall(text or ""):
        phrase = _TRAILING_CONNECTOR_RE.sub("", " ".join(match.split()))
        if len(phrase.split()) >= 2:
            found.append(phrase)
    return list(dict.fromkeys(found))[:max_terms]
