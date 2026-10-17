import unicodedata
from pathlib import Path

import pytest

from learning_agent.domain import (
    CurriculumError,
    Glossary,
    GlossaryTerm,
    Skill,
    SkillGraph,
    load_glossary,
    load_skills,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CURRICULUM = REPO_ROOT / "data" / "curriculum"

RECIPROCAL = GlossaryTerm(en="reciprocal", vi="phân số đảo ngược", forbidden_vi=("số đối",))
MIXED = GlossaryTerm(en="mixed number", vi="hỗn số", skill_id="frac.mixed")
FRACTION = GlossaryTerm(en="fraction", vi="phân số", forbidden_vi=("phân thức",))


def graph() -> SkillGraph:
    return SkillGraph(
        [
            Skill(id="frac.concept", name_vi="x", name_en="x", grade=4),
            Skill(
                id="frac.mixed", name_vi="x", name_en="x", grade=5, prerequisites=("frac.concept",)
            ),
        ]
    )


# --- lookup ------------------------------------------------------------------------------


def test_vi_for_is_case_insensitive() -> None:
    glossary = Glossary([RECIPROCAL, MIXED])
    assert glossary.vi_for("Mixed Number") == "hỗn số"


def test_vi_for_unknown_term_raises_key_error() -> None:
    with pytest.raises(KeyError):
        Glossary([MIXED]).vi_for("numerator")


def test_terms_for_skill_includes_general_terms_by_default() -> None:
    glossary = Glossary([FRACTION, MIXED, RECIPROCAL], skills=graph())
    assert [t.en for t in glossary.terms_for_skill("frac.mixed")] == [
        "fraction",
        "mixed number",
        "reciprocal",
    ]
    assert [t.en for t in glossary.terms_for_skill("frac.mixed", include_general=False)] == [
        "mixed number"
    ]


# --- forbidden translations --------------------------------------------------------------


def test_forbidden_found_reports_term_and_phrase() -> None:
    glossary = Glossary([RECIPROCAL, FRACTION])
    hits = glossary.forbidden_found("Muốn chia, ta nhân với Số Đối của phân số thứ hai.")
    assert hits == [("số đối", RECIPROCAL)]


def test_forbidden_found_matches_whole_words_only() -> None:
    # A forbidden phrase must not match when glued to letters on either side.
    glossary = Glossary([FRACTION])
    assert glossary.forbidden_found("phân thứcx") == []
    assert glossary.forbidden_found("aphân thức") == []
    assert glossary.forbidden_found("(phân thức)") == [("phân thức", FRACTION)]


def test_forbidden_found_ignores_unicode_normalization_form() -> None:
    # Same visible text, decomposed (NFD) code points: must still be found.
    glossary = Glossary([RECIPROCAL])
    text_nfd = unicodedata.normalize("NFD", "không dùng số đối ở đây")
    assert glossary.forbidden_found(text_nfd) == [("số đối", RECIPROCAL)]


def test_correct_text_has_no_forbidden_terms() -> None:
    glossary = Glossary([RECIPROCAL, FRACTION])
    assert glossary.forbidden_found("Nhân với phân số đảo ngược của phân số thứ hai.") == []


# --- invalid glossaries ------------------------------------------------------------------


def test_duplicate_english_term_is_rejected_case_insensitively() -> None:
    with pytest.raises(CurriculumError, match="duplicate glossary term: Mixed Number"):
        Glossary([MIXED, GlossaryTerm(en="Mixed Number", vi="hỗn số")])


def test_unknown_skill_id_is_rejected_when_graph_given() -> None:
    term = GlossaryTerm(en="decimal", vi="số thập phân", skill_id="frac.decimal")
    with pytest.raises(CurriculumError, match="decimal: unknown skill frac.decimal"):
        Glossary([term], skills=graph())


def test_forbidden_phrase_equal_to_translation_is_rejected() -> None:
    term = GlossaryTerm(en="fraction", vi="phân số", forbidden_vi=("Phân số",))
    with pytest.raises(CurriculumError, match="fraction: forbids its own translation"):
        Glossary([term])


# --- loading from YAML and the real data file --------------------------------------------


def test_load_glossary_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text(
        "terms:\n  - {en: mixed number, vi: hỗn số, skill_id: frac.mixed}\n", encoding="utf-8"
    )
    assert load_glossary(path, skills=graph()).vi_for("mixed number") == "hỗn số"


def test_real_glossary_is_valid_against_real_skill_graph() -> None:
    skills = load_skills(CURRICULUM / "skills.yaml")
    glossary = load_glossary(CURRICULUM / "glossary.yaml", skills=skills)
    assert glossary.vi_for("denominator") == "mẫu số"
    assert len(glossary) >= 25
