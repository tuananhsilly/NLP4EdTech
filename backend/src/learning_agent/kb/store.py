"""The knowledge base in Postgres (R3): load in-scope chunks with their skills, read them back.

Loading replaces one source's chunks as a whole, inside one transaction: the old rows are
deleted (their chunk_skill rows go with them, ON DELETE CASCADE) and the new ones inserted, so
running it twice gives the same database and a failure leaves the previous load intact.
Embeddings (R4) are written by the same INSERT, so a stored chunk always has its vector, and
the ingest command records the run in the same transaction (see kb/ingest.py).
"""

import datetime
from collections.abc import Mapping
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow, class_row
from psycopg.types.json import Jsonb

from learning_agent.domain import SkillGraph
from learning_agent.kb.chunker import Chunk
from learning_agent.kb.embedder import Vectors
from learning_agent.kb.manifest import SourceEntry

Connection = psycopg.Connection[TupleRow]


@dataclass(frozen=True)
class StoredChunk:
    id: str
    source_id: str
    module_id: str
    section_title: str
    doc_type: str
    text: str
    skills: list[str]


def load(
    conn: Connection,
    source: SourceEntry,
    skills: SkillGraph,
    chunks: list[Chunk],
    chunk_skills: Mapping[str, tuple[str, ...]],
    embeddings: Mapping[str, Vectors] | None = None,
) -> int:
    """Replace `source`'s chunks by its in-scope `chunks`; return how many were stored.

    `chunk_skills` must give the skills of every in-scope chunk (see skill_map.assign_skills);
    `embeddings`, when given, the vector of every in-scope chunk.
    """
    in_scope = [c for c in chunks if c.in_scope]
    vectors = embeddings or {}
    if embeddings is not None:
        register_vector(conn)
        missing = [c.id for c in in_scope if c.id not in embeddings]
        if missing:
            raise ValueError(f"no embedding for chunks {missing}")
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO skill (id, name_vi, name_en) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET name_vi = EXCLUDED.name_vi, name_en = EXCLUDED.name_en",
            [(s.id, s.name_vi, s.name_en) for s in map(skills.get, skills.topo_order())],
        )
        cur.execute(
            "INSERT INTO source_document "
            "(id, title, book_url, repo_url, commit_sha, license, attribution, lang) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, "
            "book_url = EXCLUDED.book_url, repo_url = EXCLUDED.repo_url, "
            "commit_sha = EXCLUDED.commit_sha, license = EXCLUDED.license, "
            "attribution = EXCLUDED.attribution, lang = EXCLUDED.lang",
            (
                source.id,
                source.title,
                source.book_url,
                source.repo_url,
                source.commit,
                source.license,
                source.attribution,
                source.language,
            ),
        )
        cur.execute("DELETE FROM chunk WHERE source_id = %s", (source.id,))
        cur.executemany(
            "INSERT INTO chunk "
            "(id, source_id, module_id, section_title, doc_type, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    c.id,
                    source.id,
                    c.module_id,
                    c.section_title,
                    c.doc_type,
                    c.text,
                    vectors.get(c.id),
                )
                for c in in_scope
            ],
        )
        cur.executemany(
            "INSERT INTO chunk_skill (chunk_id, skill_id, tagged_by) VALUES (%s, %s, 'human')",
            [(c.id, s) for c in in_scope for s in chunk_skills[c.id]],
        )
    return len(in_scope)


_SELECT_CHUNK = """
    SELECT c.id, c.source_id, c.module_id, c.section_title, c.doc_type, c.text,
           array_agg(cs.skill_id ORDER BY cs.skill_id) AS skills
    FROM chunk c JOIN chunk_skill cs ON cs.chunk_id = c.id
"""


def get_chunk(conn: Connection, chunk_id: str) -> StoredChunk | None:
    with conn.cursor(row_factory=class_row(StoredChunk)) as cur:
        cur.execute(_SELECT_CHUNK + " WHERE c.id = %s GROUP BY c.id", (chunk_id,))
        return cur.fetchone()


def count_by_skill(conn: Connection) -> dict[str, int]:
    """Number of stored chunks per skill (skills without chunks included, as 0)."""
    rows = conn.execute(
        "SELECT s.id, count(cs.chunk_id) FROM skill s "
        "LEFT JOIN chunk_skill cs ON cs.skill_id = s.id GROUP BY s.id ORDER BY s.id"
    ).fetchall()
    return {str(skill_id): int(n) for skill_id, n in rows}


def count_by_doc_type(conn: Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT doc_type, count(*) FROM chunk GROUP BY doc_type ORDER BY doc_type"
    ).fetchall()
    return {str(doc_type): int(n) for doc_type, n in rows}


def embedding_dim(conn: Connection) -> int | None:
    """Dimension of the chunk.embedding column (None before migration 0002)."""
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'chunk'::regclass AND attname = 'embedding' AND NOT attisdropped"
    ).fetchone()
    return None if row is None else int(row[0])


def nearest(conn: Connection, query: Vectors, k: int) -> list[tuple[str, float]]:
    """The k chunks closest to `query` by cosine distance (uses the HNSW index), with distances."""
    register_vector(conn)
    rows = conn.execute(
        "SELECT id, embedding <=> %(q)s AS distance FROM chunk "
        "WHERE embedding IS NOT NULL ORDER BY distance LIMIT %(k)s",
        {"q": query, "k": k},
    ).fetchall()
    return [(str(chunk_id), float(distance)) for chunk_id, distance in rows]


def record_ingest_run(
    conn: Connection,
    *,
    started_at: datetime.datetime,
    source_ids: list[str],
    git_commit: str | None,
    config: dict[str, object],
    config_hash: str,
    stats: dict[str, object],
) -> int:
    row = conn.execute(
        "INSERT INTO ingest_run (started_at, source_ids, git_commit, config, config_hash, stats) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (started_at, source_ids, git_commit, Jsonb(config), config_hash, Jsonb(stats)),
    ).fetchone()
    assert row is not None
    return int(row[0])
