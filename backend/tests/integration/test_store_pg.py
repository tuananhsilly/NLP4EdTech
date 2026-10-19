"""Loading and reading the knowledge base in a real Postgres (skipped without one)."""

from collections.abc import Iterator

import psycopg
import pytest

from learning_agent.db import connect
from learning_agent.db.migrate import migrate
from learning_agent.domain import load_skills
from learning_agent.kb import store
from learning_agent.kb.chunk import DEFAULT_MANIFEST, DEFAULT_SOURCES, build_chunks, source_dir
from learning_agent.kb.chunker import Chunk
from learning_agent.kb.embedder import HashEmbedder
from learning_agent.kb.manifest import load_manifest
from learning_agent.kb.skill_map import (
    DEFAULT_SKILL_MAP,
    DEFAULT_SKILLS,
    assign_skills,
    load_skill_map,
)

SOURCE = load_manifest(DEFAULT_MANIFEST).sources[0]
SKILLS = load_skills(DEFAULT_SKILLS)


def chunk(id_: str, text: str, *, in_scope: bool = True) -> Chunk:
    return Chunk(
        id=id_,
        module_id=id_.split(":")[0],
        section_title="S",
        doc_type="theory",
        text=text,
        in_scope=in_scope,
        scope_reason=None if in_scope else "negative_numbers",
    )


CHUNKS = [
    chunk("m1:p1", "The reciprocal of a fraction"),
    chunk("m1:p2", "Find the least common denominator"),
    chunk("m1:p3", "A negative fraction", in_scope=False),
]
CHUNK_SKILLS = {"m1:p1": ("frac.div",), "m1:p2": ("frac.common_den", "frac.add_sub.unlike_den")}


@pytest.fixture
def conn(database_url: str) -> Iterator[store.Connection]:
    with connect(database_url) as c:
        migrate(c)
        yield c
        c.execute("TRUNCATE chunk_skill, chunk, source_document, skill")


def chunk_ids(conn: store.Connection) -> list[str]:
    return [str(i) for (i,) in conn.execute("SELECT id FROM chunk ORDER BY id").fetchall()]


def test_only_in_scope_chunks_are_stored_with_their_skills(conn: store.Connection) -> None:
    assert store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS) == 2
    assert chunk_ids(conn) == ["m1:p1", "m1:p2"]
    stored = store.get_chunk(conn, "m1:p2")
    assert stored is not None
    assert stored.source_id == SOURCE.id
    assert stored.skills == ["frac.add_sub.unlike_den", "frac.common_den"]
    assert store.get_chunk(conn, "m1:p3") is None


def test_loading_twice_gives_the_same_database(conn: store.Connection) -> None:
    store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS)
    first = (chunk_ids(conn), store.count_by_skill(conn))
    store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS)
    assert (chunk_ids(conn), store.count_by_skill(conn)) == first
    assert len(store.count_by_skill(conn)) == len(SKILLS)


def test_reload_drops_chunks_that_disappeared(conn: store.Connection) -> None:
    store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS)
    store.load(conn, SOURCE, SKILLS, CHUNKS[:1], CHUNK_SKILLS)
    assert chunk_ids(conn) == ["m1:p1"]
    assert sum(store.count_by_skill(conn).values()) == 1


def test_failed_load_leaves_the_previous_load_intact(conn: store.Connection) -> None:
    store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS)
    bad = {**CHUNK_SKILLS, "m1:p1": ("frac.unknown",)}
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.load(conn, SOURCE, SKILLS, [chunk("m1:p9", "new"), *CHUNKS], {**bad, "m1:p9": ()})
    assert chunk_ids(conn) == ["m1:p1", "m1:p2"]


@pytest.mark.skipif(
    not source_dir(load_manifest(DEFAULT_MANIFEST), DEFAULT_SOURCES).exists(),
    reason="raw sources not fetched (scripts/fetch_sources.sh)",
)
def test_real_chunks_load_and_full_text_search_finds_them(conn: store.Connection) -> None:
    chunks = build_chunks(load_manifest(DEFAULT_MANIFEST), DEFAULT_SOURCES, max_words=250)
    chunk_skills = assign_skills(chunks, load_skill_map(DEFAULT_SKILL_MAP, SKILLS))
    n = store.load(conn, SOURCE, SKILLS, chunks, chunk_skills)
    assert n == sum(c.in_scope for c in chunks)
    assert all(count > 0 for count in store.count_by_skill(conn).values())
    hits = conn.execute(
        "SELECT id FROM chunk WHERE tsv @@ plainto_tsquery('english', 'least common denominators')"
    ).fetchall()
    # the glossary definition spells the term out; worked examples often write only "LCD",
    # so R5 must expand glossary aliases in the query
    assert ("m81289:fs-id2483777",) in hits


EMBEDDER = HashEmbedder(dim=1024)  # the dimension of the chunk.embedding column


def test_stored_vectors_find_their_own_chunk(conn: store.Connection) -> None:
    vectors = {c.id: EMBEDDER.embed_query(c.text) for c in CHUNKS if c.in_scope}
    store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS, vectors)
    assert store.embedding_dim(conn) == 1024
    hits = store.nearest(conn, EMBEDDER.embed_query("Find the least common denominator"), 2)
    assert hits[0][0] == "m1:p2"
    assert hits[0][1] < 1e-5  # cosine distance of a vector to itself (stored as float32)


def test_missing_embedding_is_rejected_before_writing(conn: store.Connection) -> None:
    store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS)
    with pytest.raises(ValueError, match="m1:p2"):
        store.load(conn, SOURCE, SKILLS, CHUNKS, CHUNK_SKILLS, {"m1:p1": EMBEDDER.embed_query("x")})
    assert chunk_ids(conn) == ["m1:p1", "m1:p2"]


def test_ingest_run_keeps_config_and_stats_as_json(conn: store.Connection) -> None:
    import datetime

    run_id = store.record_ingest_run(
        conn,
        started_at=datetime.datetime.now(datetime.UTC),
        source_ids=[SOURCE.id],
        git_commit="abc",
        config={"embedding": {"model_id": "hash"}},
        config_hash="0" * 64,
        stats={"chunks_stored": 2, "truncated": []},
    )
    row = conn.execute("SELECT config, stats FROM ingest_run WHERE id = %s", (run_id,)).fetchone()
    assert row == ({"embedding": {"model_id": "hash"}}, {"chunks_stored": 2, "truncated": []})
