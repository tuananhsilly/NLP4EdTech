"""English -> Vietnamese glossary of fraction terms (04 §12.1, 07 §6.2).

Used to (1) give the LLM the correct Vietnamese terms when it answers from English chunks and
(2) detect wrong translations (`forbidden_vi`) in generated Vietnamese text.
"""

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from learning_agent.domain._loading import CurriculumError, load_yaml_model
from learning_agent.domain.skills import SkillGraph, SkillId


class GlossaryTerm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    en: str = Field(min_length=1)
    vi: str = Field(min_length=1)
    skill_id: SkillId | None = None
    forbidden_vi: tuple[str, ...] = ()


class _GlossaryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[GlossaryTerm]


def _normalize(text: str) -> str:
    """NFC + casefold, so visually identical Vietnamese strings compare equal."""
    return unicodedata.normalize("NFC", text).casefold()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # (?<!\w) / (?!\w): the phrase must not be glued to letters or digits on either side.
    # In Python 3, \w is Unicode-aware, so Vietnamese letters with diacritics count as \w.
    return re.compile(r"(?<!\w)" + re.escape(_normalize(phrase)) + r"(?!\w)")


class Glossary:
    """Validated glossary. If `skills` is given, every `skill_id` must exist in it."""

    def __init__(self, terms: Iterable[GlossaryTerm], skills: SkillGraph | None = None) -> None:
        self._terms: dict[str, GlossaryTerm] = {}
        for term in terms:
            key = _normalize(term.en)
            if key in self._terms:
                raise CurriculumError(f"duplicate glossary term: {term.en}")
            if skills is not None and term.skill_id is not None and term.skill_id not in skills:
                raise CurriculumError(f"{term.en}: unknown skill {term.skill_id}")
            if any(_normalize(f) == _normalize(term.vi) for f in term.forbidden_vi):
                raise CurriculumError(f"{term.en}: forbids its own translation {term.vi!r}")
            self._terms[key] = term
        self._forbidden = [
            (phrase, term, _phrase_pattern(phrase))
            for term in self._terms.values()
            for phrase in term.forbidden_vi
        ]

    def __len__(self) -> int:
        return len(self._terms)

    def vi_for(self, en: str) -> str:
        return self._terms[_normalize(en)].vi

    def terms_for_skill(self, skill_id: str, *, include_general: bool = True) -> list[GlossaryTerm]:
        """Terms tagged with `skill_id`, plus untagged (general) terms unless excluded."""
        return [
            t
            for t in self._terms.values()
            if t.skill_id == skill_id or (include_general and t.skill_id is None)
        ]

    def forbidden_found(self, text_vi: str) -> list[tuple[str, GlossaryTerm]]:
        """(forbidden phrase, its term) for each forbidden phrase that occurs in `text_vi`."""
        text = _normalize(text_vi)
        return [(phrase, term) for phrase, term, pattern in self._forbidden if pattern.search(text)]


def load_glossary(path: Path, skills: SkillGraph | None = None) -> Glossary:
    return Glossary(load_yaml_model(path, _GlossaryFile).terms, skills=skills)
