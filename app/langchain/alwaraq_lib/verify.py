"""
Evidence verification for Ask Alwaraq answers.

The composer's JSON is checked against the passages it was given:
  - citations to unknown passage ids are removed
  - quotes that don't appear in the cited passage are removed
  - claims left without any valid citation are removed
  - each surviving claim gets a confidence label
"""

import re

from app.langchain.alwaraq_lib.normalize import normalize_for_match, tokens

QUOTE_TOKEN_COVERAGE = 0.9
_MARKER_RE = re.compile(r"\[(P\d+)\]")


def locate_quote(quote: str, passage_text: str) -> dict | None:
    """
    Check that `quote` appears in `passage_text`.
    Returns {"verbatim": bool} when found, None when not supported.
      - verbatim: exact match after normalization (diacritics, letter forms,
        punctuation, whitespace)
      - near: >= 90% of the quote's tokens appear in the passage in order-free
        form AND a contiguous run covers most of it (handles tiny OCR/spelling drift)
    """
    q = normalize_for_match(quote)
    if len(q) < 3:
        return None
    p = normalize_for_match(passage_text)
    if q in p:
        return {"verbatim": True}

    q_tokens = q.split()
    if len(q_tokens) < 4:
        return None
    p_tokens = set(p.split())
    coverage = sum(1 for t in q_tokens if t in p_tokens) / len(q_tokens)
    if coverage < QUOTE_TOKEN_COVERAGE:
        return None
    # Require a contiguous run of at least 60% of the quote to exist verbatim.
    run = max(3, int(len(q_tokens) * 0.6))
    for i in range(0, len(q_tokens) - run + 1):
        if " ".join(q_tokens[i : i + run]) in p:
            return {"verbatim": False}
    return None


def context_around(quote: str, passage_text: str, window: int = 200) -> tuple[str, str]:
    """Best-effort original-text context before/after a verified quote."""
    first = tokens(quote)[:3]
    if not first:
        return "", ""
    # Find the first quote token in the raw text (after light normalization of the raw text).
    raw = passage_text
    norm_raw = normalize_for_match(raw)
    idx_norm = norm_raw.find(" ".join(first))
    if idx_norm < 0:
        return "", ""
    # Map proportional position back onto the raw text (normalization only removes chars).
    ratio = idx_norm / max(1, len(norm_raw))
    start = int(ratio * len(raw))
    end = min(len(raw), start + len(quote))
    before = raw[max(0, start - window) : start].strip()
    after = raw[end : end + window].strip()
    return before, after


def _confidence(claim: dict, valid_citations: list[str], has_verbatim: bool) -> str:
    if claim.get("disputed"):
        return "disputed" if len(valid_citations) >= 2 else "uncertain"
    if claim.get("uncertain"):
        return "uncertain"
    if len(valid_citations) >= 2 or has_verbatim:
        return "confirmed"
    return "probable"


def strip_unknown_markers(text: str, known_labels: set[str]) -> str:
    def _sub(m):
        return m.group(0) if m.group(1) in known_labels else ""

    return re.sub(r"\s{2,}", " ", _MARKER_RE.sub(_sub, text or "")).strip()


def verify_composition(composed: dict, passages_by_label: dict[str, dict]) -> dict:
    """
    Verification for a composed piece (a hook, a blurb, a recommendation).

    The reader asked to be written for, not cited to, so a line standing on its
    own is expected here and is kept. A *quote* is still an assertion about what
    a book says, so quotes are checked exactly as strictly as anywhere else:
    anything not found in the passage it names is dropped.
    """
    known = set(passages_by_label)
    dropped = {"claims": 0, "quotes": 0, "citations": 0}
    cited: list[str] = []
    quote_info: dict[str, list[dict]] = {}
    sections_out = []

    for section in composed.get("sections") or []:
        claims_out = []
        for claim in section.get("claims") or []:
            if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
                continue
            raw_citations = [str(c).strip() for c in claim.get("citations") or []]
            valid = [c for c in dict.fromkeys(raw_citations) if c in known]
            dropped["citations"] += len(raw_citations) - len(valid)

            quotes_out = []
            for q in claim.get("quotes") or []:
                if not isinstance(q, dict):
                    continue
                label = str(q.get("passage", "")).strip()
                text = str(q.get("text", "")).strip()
                if label not in known or not text:
                    dropped["quotes"] += 1
                    continue
                found = locate_quote(text, passages_by_label[label]["text"])
                if not found:
                    dropped["quotes"] += 1
                    continue
                before, after = context_around(text, passages_by_label[label]["text"])
                quotes_out.append({"passage": label, "text": text, "verbatim": found["verbatim"]})
                quote_info.setdefault(label, []).append(
                    {
                        "text": text,
                        "verbatim": found["verbatim"],
                        "context_before": before,
                        "context_after": after,
                    }
                )
                if label not in valid:
                    valid.append(label)

            cited.extend(valid)
            claims_out.append(
                {
                    "text": strip_unknown_markers(str(claim["text"]), known),
                    "confidence": "composed",
                    "citations": valid,
                    "quotes": quotes_out,
                }
            )
        if claims_out:
            sections_out.append({"heading": str(section.get("heading") or "").strip(), "claims": claims_out})

    has_text = bool(str(composed.get("summary") or "").strip()) or bool(sections_out)
    return {
        "sections": sections_out,
        "disagreements": [],
        "cited_labels": list(dict.fromkeys(cited)),
        "quote_info": quote_info,
        "dropped": dropped,
        "has_evidence": has_text and not composed.get("no_evidence", False),
    }


def verify_answer(composed: dict, passages_by_label: dict[str, dict]) -> dict:
    """
    Returns:
      {
        "sections": [...verified sections...],
        "disagreements": [...],
        "cited_labels": [...labels actually used...],
        "quote_info": {label: [{"text", "verbatim", "context_before", "context_after"}]},
        "dropped": {"claims": n, "quotes": n, "citations": n},
        "has_evidence": bool,
      }
    """
    known = set(passages_by_label)
    dropped = {"claims": 0, "quotes": 0, "citations": 0}
    cited: list[str] = []
    quote_info: dict[str, list[dict]] = {}
    sections_out = []

    for section in composed.get("sections") or []:
        claims_out = []
        for claim in section.get("claims") or []:
            if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
                continue
            raw_citations = [str(c).strip() for c in claim.get("citations") or []]
            valid = [c for c in dict.fromkeys(raw_citations) if c in known]
            dropped["citations"] += len(raw_citations) - len(valid)

            quotes_out = []
            has_verbatim = False
            for q in claim.get("quotes") or []:
                if not isinstance(q, dict):
                    continue
                label = str(q.get("passage", "")).strip()
                text = str(q.get("text", "")).strip()
                if label not in known or not text:
                    dropped["quotes"] += 1
                    continue
                found = locate_quote(text, passages_by_label[label]["text"])
                if not found:
                    dropped["quotes"] += 1
                    continue
                before, after = context_around(text, passages_by_label[label]["text"])
                quotes_out.append({"passage": label, "text": text, "verbatim": found["verbatim"]})
                quote_info.setdefault(label, []).append(
                    {
                        "text": text,
                        "verbatim": found["verbatim"],
                        "context_before": before,
                        "context_after": after,
                    }
                )
                has_verbatim = has_verbatim or found["verbatim"]
                if label not in valid:
                    valid.append(label)

            if not valid:
                dropped["claims"] += 1
                continue

            cited.extend(valid)
            claims_out.append(
                {
                    "text": strip_unknown_markers(str(claim["text"]), known),
                    "confidence": _confidence(claim, valid, has_verbatim),
                    "citations": valid,
                    "quotes": quotes_out,
                }
            )
        if claims_out:
            sections_out.append({"heading": str(section.get("heading") or "").strip(), "claims": claims_out})

    disagreements = []
    for d in composed.get("disagreements") or []:
        if not isinstance(d, dict):
            continue
        valid = [c for c in dict.fromkeys(str(c).strip() for c in d.get("citations") or []) if c in known]
        if len(valid) >= 2:
            disagreements.append({"topic": str(d.get("topic") or "").strip(), "citations": valid})
            cited.extend(valid)

    return {
        "sections": sections_out,
        "disagreements": disagreements,
        "cited_labels": list(dict.fromkeys(cited)),
        "quote_info": quote_info,
        "dropped": dropped,
        "has_evidence": bool(sections_out) and not composed.get("no_evidence", False),
    }
