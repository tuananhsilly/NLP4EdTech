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
from typing import Literal

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
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


# Retrieval (R5). Filters apply inside each branch, before LIMIT: filtering the top-n afterwards
# could leave nothing. Ties are broken by id so that a ranking is reproducible; on a KB this
# small that makes the dense branch an exact scan instead of the approximate HNSW index.
_FILTERS = """
    AND (%(doc_type)s::text IS NULL OR c.doc_type = %(doc_type)s)
    AND (%(skill_id)s::text IS NULL OR EXISTS (
        SELECT 1 FROM chunk_skill cs WHERE cs.chunk_id = c.id AND cs.skill_id = %(skill_id)s))
"""

# OR of the query's lexemes. to_tsvector parses and stems the question exactly like chunk.tsv;
# each lexeme is quoted for tsquery input (quotes and backslashes doubled) and cast with
# ::tsquery, which does not stem again. A question with only stop words gives NULL: no match.
_OR_QUERY = """
    (SELECT string_agg('''' || replace(replace(lexeme, '\\', '\\\\'), '''', '''''') || '''', ' | ')
     FROM unnest(to_tsvector('english', %(text)s)))::tsquery
"""

_RANK_FUNCTIONS = {"ts_rank": sql.SQL("ts_rank"), "ts_rank_cd": sql.SQL("ts_rank_cd")}


@dataclass(frozen=True)
class HitRow:
    id: str
    section_title: str
    doc_type: str
    text: str
    skills: list[str]
    source_title: str
    license: str
    attribution: str


def dense_candidates(
    conn: Connection,
    query: Vectors,
    n: int,
    *,
    skill_id: str | None = None,
    doc_type: str | None = None,
) -> list[tuple[str, float]]:
    """Up to n chunks by cosine similarity to `query` (1 = same direction), highest first."""
    register_vector(conn)
    rows = conn.execute(
        "SELECT c.id, 1 - (c.embedding <=> %(q)s) AS cosine FROM chunk c "
        "WHERE c.embedding IS NOT NULL" + _FILTERS + "ORDER BY cosine DESC, c.id LIMIT %(n)s",
        {"q": query, "n": n, "skill_id": skill_id, "doc_type": doc_type},
    ).fetchall()
    return [(str(chunk_id), float(cosine)) for chunk_id, cosine in rows]


def lexical_candidates(
    conn: Connection,
    text: str,
    n: int,
    *,
    rank: Literal["ts_rank", "ts_rank_cd"],
    normalization: int,
    skill_id: str | None = None,
    doc_type: str | None = None,
) -> list[tuple[str, float]]:
    """Up to n chunks containing any word of `text` (full-text, stemmed), best rank first."""
    query = sql.SQL(
        "WITH q AS (SELECT " + _OR_QUERY + " AS query) "
        "SELECT c.id, {rank}(c.tsv, q.query, %(norm)s::int) AS score FROM chunk c, q "
        "WHERE c.tsv @@ q.query" + _FILTERS + "ORDER BY score DESC, c.id LIMIT %(n)s"
    ).format(rank=_RANK_FUNCTIONS[rank])
    rows = conn.execute(
        query,
        {"text": text, "norm": normalization, "n": n, "skill_id": skill_id, "doc_type": doc_type},
    ).fetchall()
    return [(str(chunk_id), float(score)) for chunk_id, score in rows]


def get_hits(conn: Connection, chunk_ids: list[str]) -> dict[str, HitRow]:
    """What a search result shows of each chunk, with its source's attribution (CC BY-NC-SA)."""
    with conn.cursor(row_factory=class_row(HitRow)) as cur:
        cur.execute(
            "SELECT c.id, c.section_title, c.doc_type, c.text, "
            "array_agg(cs.skill_id ORDER BY cs.skill_id) AS skills, "
            "d.title AS source_title, d.license, d.attribution "
            "FROM chunk c JOIN chunk_skill cs ON cs.chunk_id = c.id "
            "JOIN source_document d ON d.id = c.source_id "
            "WHERE c.id = ANY(%s) GROUP BY c.id, d.id",
            (chunk_ids,),
        )
        return {row.id: row for row in cur.fetchall()}


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
