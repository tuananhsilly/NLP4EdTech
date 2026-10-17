import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

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
MANIFEST = REPO_ROOT / "data" / "source_manifest.yaml"
MODULES = REPO_ROOT / "data" / "sources" / "osbooks-prealgebra-bundle" / "modules"

RECIPROCAL = GlossaryTerm(en="reciprocal", vi="phân số đảo ngược", forbidden_vi=("số đối",))
FRACTION = GlossaryTerm(en="fraction", vi="phân số", forbidden_vi=("phân thức",))
EQUIVALENT = GlossaryTerm(
    en="equivalent fractions", vi="phân số bằng nhau", noncanonical_vi=("phân số tương đương",)
)
LCD = GlossaryTerm(
    en="least common denominator",
    en_aliases=("LCD",),
    vi="mẫu số chung nhỏ nhất",
    noncanonical_vi=("mẫu số chung bé nhất",),
    skill_id="frac.mixed",
)


def graph() -> SkillGraph:
    return SkillGraph(
        [
            Skill(id="frac.concept", name_vi="x", name_en="x"),
            Skill(id="frac.mixed", name_vi="x", name_en="x", prerequisites=("frac.concept",)),
        ]
    )


# --- EN <-> VI mapping -------------------------------------------------------------------


def test_vi_for_accepts_canonical_term_and_alias_case_insensitively() -> None:
    glossary = Glossary([LCD])
    assert glossary.vi_for("Least Common Denominator") == "mẫu số chung nhỏ nhất"
    assert glossary.vi_for("lcd") == "mẫu số chung nhỏ nhất"


def test_en_for_maps_canonical_vietnamese_back_to_canonical_english() -> None:
    glossary = Glossary([LCD, FRACTION])
    assert glossary.en_for("Mẫu số chung nhỏ nhất") == "least common denominator"


def test_unknown_lookups_raise_key_error() -> None:
    glossary = Glossary([FRACTION])
    with pytest.raises(KeyError):
        glossary.vi_for("numerator")
    with pytest.raises(KeyError):
        glossary.en_for("tử số")


def test_terms_for_skill_includes_general_terms_by_default() -> None:
    glossary = Glossary([FRACTION, LCD, RECIPROCAL], skills=graph())
    assert [t.en for t in glossary.terms_for_skill("frac.mixed")] == [
        "fraction",
        "least common denominator",
        "reciprocal",
    ]
    assert [t.en for t in glossary.terms_for_skill("frac.mixed", include_general=False)] == [
        "least common denominator"
    ]


# --- forbidden (wrong meaning) -----------------------------------------------------------


def test_forbidden_found_reports_phrase_and_term() -> None:
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
    glossary = Glossary([RECIPROCAL])
    text_nfd = unicodedata.normalize("NFD", "không dùng số đối ở đây")
    assert glossary.forbidden_found(text_nfd) == [("số đối", RECIPROCAL)]


# --- non-canonical (same meaning, other wording) -----------------------------------------


def test_noncanonical_found_reports_phrase_and_term() -> None:
    glossary = Glossary([EQUIVALENT, LCD])
    assert glossary.noncanonical_found("Tìm mẫu số chung bé nhất.") == [
        ("mẫu số chung bé nhất", LCD)
    ]


def test_normalize_replaces_noncanonical_with_canonical_translation() -> None:
    glossary = Glossary([EQUIVALENT, LCD])
    assert (
        glossary.normalize("Hai phân số tương đương có cùng mẫu số chung bé nhất.")
        == "Hai phân số bằng nhau có cùng mẫu số chung nhỏ nhất."
    )


def test_normalize_keeps_sentence_initial_capital() -> None:
    glossary = Glossary([EQUIVALENT])
    assert glossary.normalize("Phân số tương đương là gì?") == "Phân số bằng nhau là gì?"


def test_normalize_leaves_canonical_and_unrelated_text_unchanged() -> None:
    glossary = Glossary([EQUIVALENT, LCD, RECIPROCAL])
    text = "Phân số bằng nhau; không đổi số đối; phân số tương đươngx."
    assert glossary.normalize(text) == text


def test_normalize_handles_decomposed_unicode_input() -> None:
    glossary = Glossary([EQUIVALENT])
    text_nfd = unicodedata.normalize("NFD", "hai phân số tương đương")
    assert glossary.normalize(text_nfd) == "hai phân số bằng nhau"


# --- invariants of a one-to-one mapping (ADR-15) -----------------------------------------


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        (
            [FRACTION, GlossaryTerm(en="Fraction", vi="khác")],
            "duplicate English term: Fraction",
        ),
        (
            [LCD, GlossaryTerm(en="lcd", vi="khác")],
            "duplicate English term: lcd",
        ),
        (
            [FRACTION, GlossaryTerm(en="fractions", vi="Phân số")],
            "Vietnamese translation used twice: Phân số",
        ),
        (
            [RECIPROCAL, GlossaryTerm(en="opposite", vi="số đối")],
            "số đối is both a canonical translation and a variant",
        ),
        (
            [EQUIVALENT, GlossaryTerm(en="x", vi="y", forbidden_vi=("phân số tương đương",))],
            "phân số tương đương is listed as a variant twice",
        ),
        (
            [GlossaryTerm(en="fraction", vi="phân số", noncanonical_vi=("Phân số",))],
            "Phân số is both a canonical translation and a variant",
        ),
    ],
)
def test_mapping_must_be_one_to_one(terms: list[GlossaryTerm], message: str) -> None:
    with pytest.raises(CurriculumError, match=re.escape(message)):
        Glossary(terms)


def test_unknown_skill_id_is_rejected_when_graph_given() -> None:
    term = GlossaryTerm(en="decimal", vi="số thập phân", skill_id="frac.decimal")
    with pytest.raises(CurriculumError, match="decimal: unknown skill frac.decimal"):
        Glossary([term], skills=graph())


# --- loading and the real data files -----------------------------------------------------


def test_load_glossary_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text(
        "terms:\n"
        "  - {en: least common denominator, en_aliases: [LCD], vi: mẫu số chung nhỏ nhất,\n"
        "     skill_id: frac.mixed}\n",
        encoding="utf-8",
    )
    assert load_glossary(path, skills=graph()).vi_for("LCD") == "mẫu số chung nhỏ nhất"


def test_real_glossary_is_valid_against_real_skill_graph() -> None:
    skills = load_skills(CURRICULUM / "skills.yaml")
    glossary = load_glossary(CURRICULUM / "glossary.yaml", skills=skills)
    assert glossary.vi_for("denominator") == "mẫu số"


def _in_scope_book_text() -> str:
    """Lowercased text of every module whose scope is not `out` (needs data/sources)."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    parts: list[str] = []
    for module in manifest["sources"][0]["modules"]:
        if module["scope"] == "out":
            continue
        root = ET.parse(MODULES / module["id"] / "index.cnxml").getroot()
        parts.append(" ".join("".join(root.itertext()).split()))
    return " ".join(parts).lower()


@pytest.mark.skipif(
    not MODULES.exists(), reason="raw sources not fetched (scripts/fetch_sources.sh)"
)
def test_every_english_term_occurs_in_prealgebra() -> None:
    # ADR-15: English terminology comes only from the KB source. Plural "s" is accepted.
    text = _in_scope_book_text()
    glossary = load_glossary(CURRICULUM / "glossary.yaml")
    missing = [
        en
        for term in glossary.terms()
        for en in (term.en, *term.en_aliases)
        if not re.search(r"(?<!\w)" + re.escape(en.lower()) + r"s?(?!\w)", text)
    ]
    assert missing == []
