import pytest

from learning_agent.kb.query import PreparedQuery
from learning_agent.kb.retrieval import Hit, SearchResult
from learning_agent.kb.search import format_result, main

QUERY = PreparedQuery(
    original="Số nghịch đảo của phân số là gì?",
    normalized="Số nghịch đảo của phân số là gì?",
    en_terms=("reciprocal", "fraction"),
    forbidden=(),
    dense_text="Số nghịch đảo của phân số là gì? reciprocal fraction",
    lexical_text="Số nghịch đảo của phân số là gì? reciprocal fraction",
)


def hit(
    chunk_id: str, *, dense: tuple[int, float] | None, lexical: tuple[int, float] | None
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text="The reciprocal of a fraction is found by inverting the fraction.\nSecond line.",
        section_title="Divide Fractions",
        doc_type="definition",
        skills=["frac.div"],
        source_title="Prealgebra 2e",
        license="CC BY-NC-SA 4.0",
        attribution="OpenStax",
        rrf_score=0.0328,
        dense_rank=dense[0] if dense else None,
        dense_cosine=dense[1] if dense else None,
        lexical_rank=lexical[0] if lexical else None,
        lexical_score=lexical[1] if lexical else None,
    )


def test_format_shows_query_preparation_threshold_and_branch_ranks() -> None:
    result = SearchResult(
        QUERY,
        [
            hit("m1:a", dense=(1, 0.712), lexical=(3, 0.022)),
            hit("m1:b", dense=None, lexical=(1, 0.05)),
        ],
        top_cosine=0.712,
        tau=0.5,
        refused=False,
    )
    out = format_result(result)
    assert "en_terms:   reciprocal, fraction" in out
    assert "forbidden:  -" in out
    assert "top cosine: 0.712 (tau 0.500) -> answer" in out
    lines = out.splitlines()
    row_a = next(line for line in lines if "m1:a" in line)
    assert "1 0.712" in row_a and "3 0.022" in row_a
    row_b = next(line for line in lines if "m1:b" in line)
    assert "-" in row_b.split("m1:b")[0]  # no dense rank
    # the preview is one line (newlines in the chunk text flattened), cut at 70 characters
    assert row_a.endswith("inverting the fraction. Secon")


def test_format_explains_a_refusal() -> None:
    below = SearchResult(QUERY, [], top_cosine=0.31, tau=0.5, refused=True)
    assert "top cosine: 0.310 (tau 0.500) -> REFUSED: below tau" in format_result(below)
    empty = SearchResult(QUERY, [], top_cosine=None, tau=0.5, refused=True)
    assert "REFUSED: no chunk matches the filters" in format_result(empty)


def test_unreachable_database_fails_fast_with_a_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Port 1 refuses at once; the model must not be loaded before the database is checked.
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none?connect_timeout=2")
    assert main(["phân số là gì?"]) == 1
    assert "docker compose up -d" in capsys.readouterr().err
