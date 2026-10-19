"""The knowledge base in Postgres (R3): load in-scope chunks with their skills, read them back.

Loading replaces one source's chunks as a whole, inside one transaction: the old rows are
deleted (their chunk_skill rows go with them, ON DELETE CASCADE) and the new ones inserted, so
running it twice gives the same database and a failure leaves the previous load intact.
Embeddings (R4) are stored on the chunk rows, so they must be recomputed after every load.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow, class_row

from learning_agent.domain import SkillGraph
from learning_agent.kb.chunker import Chunk
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
) -> int:
    """Replace `source`'s chunks by its in-scope `chunks`; return how many were stored.

    `chunk_skills` must give the skills of every in-scope chunk (see skill_map.assign_skills).
    """
    in_scope = [c for c in chunks if c.in_scope]
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
            "INSERT INTO chunk (id, source_id, module_id, section_title, doc_type, text) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(c.id, source.id, c.module_id, c.section_title, c.doc_type, c.text) for c in in_scope],
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
