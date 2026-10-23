"""Query preparation for retrieval (R5, 07 §11 step 1): deterministic, no LLM.

The KB is English and BGE-M3 does not know some Vietnamese terms (R4: "số nghịch đảo"), so a
question is mapped onto the KB's vocabulary with the glossary (ADR-15):

1. `Glossary.normalize` rewrites non-canonical Vietnamese wordings to the canonical term;
2. spaced fractions are joined ("7 / 12" -> "7/12": Postgres keeps "7/12" as one token);
3. `Glossary.terms_found` lists the glossary terms in the question, and their English term and
   aliases are appended (query expansion), except those the question already contains.

Forbidden wordings ("số đối" for reciprocal) are reported, never mapped: the student may really
mean another concept, and the relevance threshold decides whether to answer.
"""

import re
from dataclasses import dataclass
from typing import Literal

from learning_agent.domain import Glossary

# Text embedded for the dense branch (R5-D2): a = normalized question; b = normalized question
# + English terms; c = English terms only (the normalized question when none was found).
DenseMode = Literal["a", "b", "c"]

_SPACED_FRACTION = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


@dataclass(frozen=True)
class PreparedQuery:
    original: str
    normalized: str
    en_terms: tuple[str, ...]  # English terms and aliases of the glossary terms found
    forbidden: tuple[str, ...]  # forbidden wordings found (diagnostic only)
    dense_text: str
    lexical_text: str  # always the normalized question + the English terms it lacks


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def prepare_query(
    text: str, glossary: Glossary, *, dense_mode: DenseMode, max_chars: int
) -> PreparedQuery:
    """Prepare `text` for both retrieval branches; ValueError if it is empty or too long."""
    if not text.strip():
        raise ValueError("the query is empty")
    if len(text) > max_chars:
        raise ValueError(f"the query is longer than {max_chars} characters")

    normalized = glossary.normalize(text)
    normalized = _SPACED_FRACTION.sub(r"\1/\2", normalized)
    normalized = " ".join(normalized.split())

    en_terms = tuple(
        dict.fromkeys(
            en for term in glossary.terms_found(normalized) for en in (term.en, *term.en_aliases)
        )
    )
    missing = [en for en in en_terms if not _contains_phrase(normalized, en)]
    expanded = " ".join([normalized, *missing])
    dense_text = {
        "a": normalized,
        "b": expanded,
        "c": " ".join(en_terms) or normalized,
    }[dense_mode]
    return PreparedQuery(
        original=text,
        normalized=normalized,
        en_terms=en_terms,
        forbidden=tuple(
            dict.fromkeys(phrase for phrase, _ in glossary.forbidden_found(normalized))
        ),
        dense_text=dense_text,
        lexical_text=expanded,
    )
