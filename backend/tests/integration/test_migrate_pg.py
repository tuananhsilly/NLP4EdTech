"""Migrations against a real Postgres (skipped without one; see conftest.py)."""

import psycopg
import pytest

from learning_agent.db import connect
from learning_agent.db.migrate import migrate


def test_first_run_creates_the_kb_tables_and_second_run_does_nothing(database_url: str) -> None:
    with connect(database_url) as conn:
        assert [m.name for m in migrate(conn)] == ["0001_kb.sql", "0002_embedding.sql"]
        assert migrate(conn) == []
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
        extension = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        assert {t for (t,) in tables} >= {
            "source_document",
            "skill",
            "chunk",
            "chunk_skill",
            "ingest_run",
        }
        assert extension.fetchone() is not None


def test_embedding_column_has_the_model_dimension_and_an_hnsw_index(database_url: str) -> None:
    with connect(database_url) as conn:
        migrate(conn)
        column_type = conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunk'::regclass AND attname = 'embedding'"
        ).fetchone()
        index = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunk_embedding_idx'"
        ).fetchone()
        assert column_type == ("vector(1024)",)
        assert index is not None
        assert "hnsw (embedding vector_cosine_ops)" in str(index[0])


def test_tsv_is_generated_with_english_stemming(database_url: str) -> None:
    # "fractions" and "fraction" share a stem. Not every verb form does: Snowball stems
    # "adding" to 'ad' but "add" to 'add' (to handle in R5 query normalization).
    with connect(database_url) as conn:
        migrate(conn)
        with conn.transaction(force_rollback=True):
            conn.execute(
                "INSERT INTO source_document VALUES ('s', 't', 'u', 'r', 'c', 'l', 'a', 'en')"
            )
            conn.execute(
                "INSERT INTO chunk (id, source_id, module_id, section_title, doc_type, text) "
                "VALUES ('m1:p1', 's', 'm1', 'S', 'theory', 'Adding fractions with like terms')"
            )
            row = conn.execute(
                "SELECT id FROM chunk WHERE tsv @@ plainto_tsquery('english', 'a fraction')"
            ).fetchone()
            assert row == ("m1:p1",)


def test_unknown_doc_type_is_rejected(database_url: str) -> None:
    with connect(database_url) as conn:
        migrate(conn)
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction(force_rollback=True):
            conn.execute("INSERT INTO source_document VALUES ('s','t','u','r','c','l','a','en')")
            conn.execute(
                "INSERT INTO chunk (id, source_id, module_id, section_title, doc_type, text) "
                "VALUES ('m1:x', 's', 'm1', 'S', 'exercise', 'x')"
            )
