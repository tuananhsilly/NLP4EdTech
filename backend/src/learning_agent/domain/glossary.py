"""The project's shared English <-> Vietnamese term mapping, the "translator" of ADR-15.

- English terms (and aliases such as "LCD") come only from OpenStax Prealgebra 2e, the KB source.
- Each English term has exactly one canonical Vietnamese translation, and no two terms share
  one, so the mapping works in both directions: EN -> VI when answering from English chunks,
  VI -> EN when matching a Vietnamese query against English text.
- Known wrong wordings come in two levels:
  `forbidden_vi`    wrong meaning (e.g. "số đối" for reciprocal): callers must block the text;
  `noncanonical_vi` right meaning, other wording: `normalize()` rewrites it to the canonical term.
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
    en_aliases: tuple[str, ...] = ()
    vi: str = Field(min_length=1)
    skill_id: SkillId | None = None
    forbidden_vi: tuple[str, ...] = ()
    noncanonical_vi: tuple[str, ...] = ()


class _GlossaryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[GlossaryTerm]


def _key(text: str) -> str:
    """Lookup key: NFC + casefold, so visually identical strings compare equal."""
    return unicodedata.normalize("NFC", text).casefold()


_EN_PLURAL = r"(?:e?s)?"  # "denominators", "boxes": English terms are matched with their plural


def _phrases_regex(phrases: Iterable[str], suffix: str = "") -> re.Pattern[str] | None:
    """One regex matching any of `phrases` as a whole phrase, case-insensitively.

    (?<!\\w) / (?!\\w) stop a match inside a longer word; \\w is Unicode-aware, so Vietnamese
    letters with diacritics count. Longer phrases come first so they win over their prefixes.
    An optional `suffix` pattern may follow the phrase; group "phrase" is the phrase without it.
    """
    alternatives = sorted({unicodedata.normalize("NFC", p) for p in phrases}, key=len, reverse=True)
    if not alternatives:
        return None
    body = "|".join(re.escape(p) for p in alternatives)
    return re.compile(rf"(?<!\w)(?P<phrase>{body}){suffix}(?!\w)", re.IGNORECASE)


class Glossary:
    """Validated one-to-one EN <-> VI mapping. If `skills` is given, every `skill_id` must exist."""

    def __init__(self, terms: Iterable[GlossaryTerm], skills: SkillGraph | None = None) -> None:
        self._terms: list[GlossaryTerm] = []
        self._by_en: dict[str, GlossaryTerm] = {}
        self._by_vi: dict[str, GlossaryTerm] = {}
        for term in terms:
            for en in (term.en, *term.en_aliases):
                if _key(en) in self._by_en:
                    raise CurriculumError(f"duplicate English term: {en}")
                self._by_en[_key(en)] = term
            if _key(term.vi) in self._by_vi:
                raise CurriculumError(f"Vietnamese translation used twice: {term.vi}")
            self._by_vi[_key(term.vi)] = term
            if skills is not None and term.skill_id is not None and term.skill_id not in skills:
                raise CurriculumError(f"{term.en}: unknown skill {term.skill_id}")
            self._terms.append(term)

        # Variants are checked after all canonical translations are known, so that a variant
        # of one term can never shadow the canonical translation of another.
        self._forbidden: list[tuple[str, GlossaryTerm]] = []
        self._noncanonical: dict[str, GlossaryTerm] = {}
        seen_variants: set[str] = set()
        for term in self._terms:
            for phrase in (*term.forbidden_vi, *term.noncanonical_vi):
                if _key(phrase) in self._by_vi:
                    raise CurriculumError(f"{phrase} is both a canonical translation and a variant")
                if _key(phrase) in seen_variants:
                    raise CurriculumError(f"{phrase} is listed as a variant twice")
                seen_variants.add(_key(phrase))
            self._forbidden.extend((phrase, term) for phrase in term.forbidden_vi)
            self._noncanonical.update((_key(phrase), term) for phrase in term.noncanonical_vi)

        self._forbidden_re = _phrases_regex(phrase for phrase, _ in self._forbidden)
        self._noncanonical_re = _phrases_regex(self._noncanonical)
        self._vi_re = _phrases_regex(term.vi for term in self._terms)
        self._en_re = _phrases_regex(
            (en for term in self._terms for en in (term.en, *term.en_aliases)), _EN_PLURAL
        )

    def __len__(self) -> int:
        return len(self._terms)

    def terms(self) -> list[GlossaryTerm]:
        return list(self._terms)

    def vi_for(self, en: str) -> str:
        """Canonical Vietnamese for an English term or alias (case-insensitive)."""
        return self._by_en[_key(en)].vi

    def en_for(self, vi: str) -> str:
        """Canonical English term for a canonical Vietnamese translation."""
        return self._by_vi[_key(vi)].en

    def terms_for_skill(self, skill_id: str, *, include_general: bool = True) -> list[GlossaryTerm]:
        """Terms tagged with `skill_id`, plus untagged (general) terms unless excluded."""
        return [
            t
            for t in self._terms
            if t.skill_id == skill_id or (include_general and t.skill_id is None)
        ]

    def forbidden_found(self, text_vi: str) -> list[tuple[str, GlossaryTerm]]:
        """(forbidden phrase, its term) for each wrong-meaning phrase that occurs in `text_vi`."""
        found = self._matched_keys(self._forbidden_re, text_vi)
        return [(phrase, term) for phrase, term in self._forbidden if _key(phrase) in found]

    def noncanonical_found(self, text_vi: str) -> list[tuple[str, GlossaryTerm]]:
        """(non-canonical phrase, its term) for each such phrase that occurs in `text_vi`."""
        found = self._matched_keys(self._noncanonical_re, text_vi)
        return [
            (phrase, term)
            for term in self._terms
            for phrase in term.noncanonical_vi
            if _key(phrase) in found
        ]

    def terms_found(self, text: str) -> list[GlossaryTerm]:
        """Terms whose canonical Vietnamese, English term or alias occurs in `text`.

        In order of first appearance, without duplicates. English matches accept a plural
        ending. Only canonical wordings count: call `normalize()` first so that non-canonical
        Vietnamese becomes canonical; forbidden phrases are never terms.
        """
        text = unicodedata.normalize("NFC", text)
        matches: list[tuple[int, GlossaryTerm]] = []
        for pattern, by_key in ((self._vi_re, self._by_vi), (self._en_re, self._by_en)):
            if pattern is not None:
                matches.extend(
                    (m.start(), by_key[_key(m.group("phrase"))]) for m in pattern.finditer(text)
                )
        found: list[GlossaryTerm] = []
        for _, term in sorted(matches, key=lambda pair: pair[0]):
            if term not in found:
                found.append(term)
        return found

    def normalize(self, text_vi: str) -> str:
        """Return `text_vi` (as NFC) with every non-canonical phrase replaced by its canonical term.

        A replacement keeps an upper-case first letter (sentence start); forbidden phrases are
        left untouched because they need the caller to block the text, not a silent rewrite.
        """
        text = unicodedata.normalize("NFC", text_vi)
        if self._noncanonical_re is None:
            return text

        def replace(match: re.Match[str]) -> str:
            canonical = self._noncanonical[_key(match.group())].vi
            if match.group()[0].isupper():
                canonical = canonical[0].upper() + canonical[1:]
            return canonical

        return self._noncanonical_re.sub(replace, text)

    @staticmethod
    def _matched_keys(pattern: re.Pattern[str] | None, text_vi: str) -> set[str]:
        if pattern is None:
            return set()
        return {_key(m.group()) for m in pattern.finditer(unicodedata.normalize("NFC", text_vi))}


def load_glossary(path: Path, skills: SkillGraph | None = None) -> Glossary:
    return Glossary(load_yaml_model(path, _GlossaryFile).terms, skills=skills)
