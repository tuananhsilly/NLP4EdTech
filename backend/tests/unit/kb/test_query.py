import pytest

from learning_agent.domain import Glossary, GlossaryTerm, load_glossary
from learning_agent.kb.query import DenseMode, PreparedQuery, prepare_query
from learning_agent.kb.skill_map import DEFAULT_GLOSSARY

FRACTION = GlossaryTerm(en="fraction", vi="phân số")
RECIPROCAL = GlossaryTerm(
    en="reciprocal",
    vi="số nghịch đảo",
    forbidden_vi=("số đối",),
    noncanonical_vi=("phân số đảo ngược",),
)
LCD = GlossaryTerm(
    en="least common denominator",
    en_aliases=("LCD",),
    vi="mẫu số chung nhỏ nhất",
    noncanonical_vi=("mẫu số chung bé nhất",),
)
GLOSSARY = Glossary([FRACTION, RECIPROCAL, LCD])


def prepare(text: str, mode: DenseMode = "b", max_chars: int = 1000) -> PreparedQuery:
    return prepare_query(text, GLOSSARY, dense_mode=mode, max_chars=max_chars)


def test_vietnamese_query_is_normalized_and_expanded_with_english_terms_and_aliases() -> None:
    q = prepare("Tìm mẫu số chung bé nhất")
    assert q.original == "Tìm mẫu số chung bé nhất"
    assert q.normalized == "Tìm mẫu số chung nhỏ nhất"
    assert q.en_terms == ("least common denominator", "LCD")
    assert q.dense_text == "Tìm mẫu số chung nhỏ nhất least common denominator LCD"
    assert q.lexical_text == q.dense_text


def test_spaced_fractions_are_joined() -> None:
    # Postgres keeps "7/12" as one token but splits "7 / 12" (R3, R5 step 0).
    q = prepare("Rút gọn 7 / 12 và 5/ 6, hỗn số 2 3/4")
    assert q.normalized == "Rút gọn 7/12 và 5/6, hỗn số 2 3/4"


def test_whitespace_is_collapsed() -> None:
    assert prepare("  phân   số\n là gì ").normalized == "phân số là gì"


@pytest.mark.parametrize(
    ("mode", "dense_text"),
    [
        ("a", "Số nghịch đảo của phân số là gì?"),
        ("b", "Số nghịch đảo của phân số là gì? reciprocal fraction"),
        ("c", "reciprocal fraction"),
    ],
)
def test_dense_text_follows_the_mode(mode: DenseMode, dense_text: str) -> None:
    q = prepare("Phân số đảo ngược của phân số là gì?", mode)
    assert q.dense_text == dense_text
    # The lexical branch always gets the expanded query, whatever the dense mode.
    assert q.lexical_text == "Số nghịch đảo của phân số là gì? reciprocal fraction"


def test_mode_c_without_terms_falls_back_to_the_normalized_query() -> None:
    assert prepare("Thủ đô của Pháp là gì?", "c").dense_text == "Thủ đô của Pháp là gì?"


def test_terms_already_in_an_english_query_are_not_repeated() -> None:
    q = prepare("What is the least common denominator of two fractions?")
    assert q.en_terms == ("least common denominator", "LCD", "fraction")
    # Only whole phrases count: "fractions" does not hide "fraction", so it is added.
    assert q.dense_text == "What is the least common denominator of two fractions? LCD fraction"


def test_forbidden_phrases_are_reported_but_not_expanded() -> None:
    q = prepare("Số đối của 3/4 là gì?")
    assert q.forbidden == ("số đối",)
    assert q.en_terms == ()
    assert q.dense_text == "Số đối của 3/4 là gì?"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare(text)


def test_query_longer_than_the_limit_is_rejected() -> None:
    assert prepare("x" * 10, max_chars=10).normalized == "x" * 10
    with pytest.raises(ValueError, match="longer than 10"):
        prepare("x" * 11, max_chars=10)


def test_real_glossary_expands_the_r4_failure_case() -> None:
    # R4: BGE-M3 alone found 0/5 relevant chunks for this question.
    glossary = load_glossary(DEFAULT_GLOSSARY)
    q = prepare_query("Số nghịch đảo của phân số là gì?", glossary, dense_mode="b", max_chars=1000)
    assert q.en_terms == ("reciprocal", "fraction")
