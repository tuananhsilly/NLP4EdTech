"""Smoke checks of retrieval on the real KB with the real model (R5, spec criteria 11-13).

These read the database filled by `kb.ingest` (never write to it) and load BGE-M3 on the GPU.
They are skipped when the database is down, the KB is not ingested or the model is not cached.
They guard known cases; they are not evaluation numbers (R6 measures Recall@k and MRR), and the
refusal checks depend on the provisional tau in config/default.yaml.
"""

from collections.abc import Iterator

import psycopg
import pytest

from learning_agent.config import DEFAULT_CONFIG, Settings, load_settings
from learning_agent.db import connect
from learning_agent.domain import load_glossary, load_skills
from learning_agent.kb import store
from learning_agent.kb.embedder import SentenceTransformerEmbedder
from learning_agent.kb.retrieval import SearchResult, search
from learning_agent.kb.skill_map import DEFAULT_GLOSSARY, DEFAULT_SKILLS

SETTINGS: Settings = load_settings(DEFAULT_CONFIG)
SKILLS = load_skills(DEFAULT_SKILLS)
GLOSSARY = load_glossary(DEFAULT_GLOSSARY, SKILLS)
RECIPROCAL_DEFINITION = "m81286:fs-id2946118"


def _model_is_cached() -> bool:  # same check as tests/unit/kb/test_embedder.py
    from huggingface_hub import try_to_load_from_cache

    s = SETTINGS.embedding
    return isinstance(try_to_load_from_cache(s.model_id, "config.json", revision=s.revision), str)


pytestmark = pytest.mark.skipif(
    not _model_is_cached(), reason="model not downloaded (run kb.ingest once)"
)


@pytest.fixture(scope="module")
def conn() -> Iterator[store.Connection]:
    try:
        c = connect()
    except (RuntimeError, psycopg.OperationalError) as e:
        pytest.skip(f"Postgres is not reachable ({e}); run `docker compose up -d`")
    with c:
        try:
            embedded = c.execute("SELECT count(embedding) FROM chunk").fetchone()
        except psycopg.errors.UndefinedTable:
            pytest.skip("the KB is not ingested (run kb.ingest)")
        if embedded is None or embedded[0] == 0:
            pytest.skip("the KB has no embeddings (run kb.ingest)")
        yield c


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(SETTINGS.embedding)


def run(conn: store.Connection, embedder: SentenceTransformerEmbedder, text: str) -> SearchResult:
    return search(conn, embedder, GLOSSARY, SKILLS, SETTINGS.retrieval, text)


def test_vietnamese_reciprocal_question_finds_the_definition(
    conn: store.Connection, embedder: SentenceTransformerEmbedder
) -> None:
    # R4: the model alone found 0/5 relevant chunks; the glossary adds "reciprocal fraction".
    result = run(conn, embedder, "Số nghịch đảo của phân số là gì?")
    assert not result.refused
    assert RECIPROCAL_DEFINITION in [h.chunk_id for h in result.hits]


def test_english_reciprocal_question_still_ranks_the_definition_first(
    conn: store.Connection, embedder: SentenceTransformerEmbedder
) -> None:
    result = run(conn, embedder, "What is the reciprocal of a fraction?")
    assert result.hits[0].chunk_id == RECIPROCAL_DEFINITION


@pytest.mark.parametrize("text", ["Thủ đô của Pháp là gì?", "What is the weather today?"])
def test_unrelated_questions_are_refused(
    conn: store.Connection, embedder: SentenceTransformerEmbedder, text: str
) -> None:
    result = run(conn, embedder, text)
    assert result.refused
    assert result.hits == []
