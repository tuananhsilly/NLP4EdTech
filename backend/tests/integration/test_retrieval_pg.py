"""Candidate queries of the two retrieval branches in a real Postgres (skipped without one).

HashEmbedder vectors carry no meaning: dense tests only use "a text finds its own chunk".
"""

from collections.abc import Iterator

import pytest

from learning_agent.db import connect
from learning_agent.db.migrate import migrate
from learning_agent.domain import load_skills
from learning_agent.kb import store
from learning_agent.kb.chunk import DEFAULT_MANIFEST
from learning_agent.kb.chunker import Chunk, DocType
from learning_agent.kb.embedder import HashEmbedder
from learning_agent.kb.manifest import load_manifest
from learning_agent.kb.skill_map import DEFAULT_SKILLS

SOURCE = load_manifest(DEFAULT_MANIFEST).sources[0]
SKILLS = load_skills(DEFAULT_SKILLS)
EMBEDDER = HashEmbedder(dim=1024)


def chunk(id_: str, doc_type: DocType, text: str) -> Chunk:
    return Chunk(
        id=id_,
        module_id="m1",
        section_title=f"Section of {id_}",
        doc_type=doc_type,
        text=text,
        in_scope=True,
        scope_reason=None,
    )


CHUNKS = [
    chunk("m1:def", "definition", "The reciprocal of a fraction is found by inverting it."),
    chunk("m1:lcd", "worked_example", "Find the LCD of 1/4 and 1/6."),
    chunk(
        "m1:add",
        "theory",
        "To add fractions with different denominators, rewrite them with a common denominator.",
    ),
    chunk("m1:mul", "theory", "Multiply the numerators and multiply the denominators."),
]
CHUNK_SKILLS = {
    "m1:def": ("frac.div",),
    "m1:lcd": ("frac.common_den",),
    "m1:add": ("frac.add_sub.unlike_den", "frac.common_den"),
    "m1:mul": ("frac.mul",),
}
TEXT = {c.id: c.text for c in CHUNKS}


@pytest.fixture(scope="module")
def conn(database_url: str) -> Iterator[store.Connection]:
    with connect(database_url) as c:
        migrate(c)
        vectors = {c.id: EMBEDDER.embed_query(c.text) for c in CHUNKS}
        store.load(c, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS, vectors)
        yield c


def lexical(conn: store.Connection, text: str, **filters: str) -> list[str]:
    hits = store.lexical_candidates(conn, text, 20, rank="ts_rank", normalization=1, **filters)
    return [chunk_id for chunk_id, _ in hits]


def dense(conn: store.Connection, text: str, **filters: str) -> list[tuple[str, float]]:
    return store.dense_candidates(conn, EMBEDDER.embed_query(text), 20, **filters)


# --- lexical branch ----------------------------------------------------------------------


def test_lexical_matches_any_word_not_all_words(conn: store.Connection) -> None:
    # plainto_tsquery would AND every word; "quickly" and "unlike" occur in no chunk.
    ids = lexical(conn, "How can I add two fractions with unlike denominators quickly?")
    assert ids[0] == "m1:add"  # add + fraction + denominator
    assert "m1:mul" in ids  # denominator only


def test_lexical_scores_are_positive_and_descending(conn: store.Connection) -> None:
    hits = store.lexical_candidates(
        conn, "fraction denominators", 20, rank="ts_rank", normalization=1
    )
    scores = [score for _, score in hits]
    assert scores and all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_lexical_keeps_fractions_as_tokens(conn: store.Connection) -> None:
    assert lexical(conn, "What is 1/4?") == ["m1:lcd"]


@pytest.mark.parametrize(
    "text",
    ["it's 3/4 & (5/6) | !x: y* 'z'", r"C:\path a\b O'Neil", "':*&|!()<->", "a <-> b <2> c"],
)
def test_lexical_never_fails_on_tsquery_syntax_in_user_text(
    conn: store.Connection, text: str
) -> None:
    lexical(conn, text)  # must not raise


@pytest.mark.parametrize("text", ["what is the", "là gì", "zebra"])
def test_lexical_without_matching_words_returns_nothing(conn: store.Connection, text: str) -> None:
    assert lexical(conn, text) == []


def test_lexical_respects_the_limit(conn: store.Connection) -> None:
    hits = store.lexical_candidates(
        conn, "fraction denominators reciprocal LCD", 2, rank="ts_rank", normalization=1
    )
    assert len(hits) == 2


def test_lexical_rank_function_is_selectable(conn: store.Connection) -> None:
    hits = store.lexical_candidates(conn, "add fractions", 20, rank="ts_rank_cd", normalization=0)
    assert hits[0][0] == "m1:add"


# --- dense branch ------------------------------------------------------------------------


def test_dense_returns_cosine_similarity(conn: store.Connection) -> None:
    hits = dense(conn, TEXT["m1:mul"])
    assert hits[0][0] == "m1:mul"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == len(CHUNKS)
    assert all(-1.0 <= cosine <= 1.0 for _, cosine in hits)


# --- filters (both branches) -------------------------------------------------------------

QUERY = "fraction denominators reciprocal LCD 1/4"


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"doc_type": "theory"}, {"m1:add", "m1:mul"}),
        ({"skill_id": "frac.common_den"}, {"m1:lcd", "m1:add"}),
        ({"skill_id": "frac.common_den", "doc_type": "theory"}, {"m1:add"}),
        ({"skill_id": "frac.div"}, {"m1:def"}),
        ({"doc_type": "procedure"}, set[str]()),
    ],
)
def test_filters_restrict_both_branches(
    conn: store.Connection, filters: dict[str, str], expected: set[str]
) -> None:
    assert set(lexical(conn, QUERY, **filters)) == expected
    assert {chunk_id for chunk_id, _ in dense(conn, TEXT["m1:mul"], **filters)} == expected


def test_filters_apply_before_the_limit(conn: store.Connection) -> None:
    # The query vector is m1:mul's, so m1:def is far down the unfiltered ranking;
    # filtering after LIMIT 1 would return nothing.
    hits = store.dense_candidates(
        conn, EMBEDDER.embed_query(TEXT["m1:mul"]), 1, skill_id="frac.div"
    )
    assert [chunk_id for chunk_id, _ in hits] == ["m1:def"]


# --- hits --------------------------------------------------------------------------------


def test_get_hits_returns_chunk_skills_and_attribution(conn: store.Connection) -> None:
    hits = store.get_hits(conn, ["m1:add", "m1:def", "m1:unknown"])
    assert set(hits) == {"m1:add", "m1:def"}
    add = hits["m1:add"]
    assert add.text == TEXT["m1:add"]
    assert add.section_title == "Section of m1:add"
    assert add.doc_type == "theory"
    assert add.skills == ["frac.add_sub.unlike_den", "frac.common_den"]
    assert add.source_title == SOURCE.title
    assert add.license == SOURCE.license
    assert add.attribution == SOURCE.attribution


def test_get_hits_of_nothing_is_empty(conn: store.Connection) -> None:
    assert store.get_hits(conn, []) == {}
