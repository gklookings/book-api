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
