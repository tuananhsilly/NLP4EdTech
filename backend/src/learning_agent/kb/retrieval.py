"""Hybrid retrieval over the knowledge base (R5, 07 §11).

    question -> prepare_query (glossary) -> dense branch (pgvector) + lexical branch (full text)
             -> Reciprocal Rank Fusion -> top k -> threshold tau -> hits with attribution

Ranking and refusing are separate: RRF orders chunks by rank only, so its score says nothing
about relevance (the first chunk always gets about 1/61). Refusal therefore uses an absolute
signal, the cosine similarity of the best dense candidate (R5-D1); when it is below tau no hit
is returned at all (R5-D7), so the answer step cannot use chunks under the threshold.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from learning_agent.config import RetrievalSettings
from learning_agent.domain import Glossary, SkillGraph
from learning_agent.kb import store
from learning_agent.kb.chunker import DocType
from learning_agent.kb.embedder import Embedder
from learning_agent.kb.query import DenseMode, PreparedQuery, prepare_query

Branches = Literal["hybrid", "dense", "lexical"]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    section_title: str
    doc_type: str
    skills: list[str]
    source_title: str
    license: str
    attribution: str
    rrf_score: float
    dense_rank: int | None  # None: not among the dense candidates (or branch not used)
    dense_cosine: float | None
    lexical_rank: int | None
    lexical_score: float | None


@dataclass(frozen=True)
class SearchResult:
    query: PreparedQuery
    hits: list[Hit]  # empty when refused
    top_cosine: float | None  # best dense cosine; None when the filters leave no chunk
    tau: float
    refused: bool


def rrf_fuse(rankings: Sequence[Sequence[str]], rrf_k: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (rrf_k + rank), rank from 1.

    Best first; equal scores are ordered by chunk id so that the result is reproducible.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def search(
    conn: store.Connection,
    embedder: Embedder,
    glossary: Glossary,
    skills: SkillGraph,
    settings: RetrievalSettings,
    text: str,
    *,
    k: int | None = None,
    skill_id: str | None = None,
    doc_type: str | None = None,
    dense_mode: DenseMode | None = None,
    branches: Branches | None = None,
    apply_threshold: bool = True,
) -> SearchResult:
    """Search the KB for `text`. Options left as None come from `settings`.

    ValueError for an empty or too long question, an unknown skill or doc_type, or k < 1.
    """
    k = settings.k if k is None else k
    branches = branches or settings.branches
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if skill_id is not None and skill_id not in skills:
        raise ValueError(f"unknown skill_id: {skill_id}")
    if doc_type is not None and doc_type not in get_args(DocType):
        raise ValueError(f"unknown doc_type: {doc_type}")
    query = prepare_query(
        text,
        glossary,
        dense_mode=dense_mode or settings.dense_query,
        max_chars=settings.max_query_chars,
    )

    n = settings.candidates_per_branch
    # The dense branch always runs: the threshold needs its best cosine, whatever the branches.
    dense = store.dense_candidates(
        conn, embedder.embed_query(query.dense_text), n, skill_id=skill_id, doc_type=doc_type
    )
    lexical = (
        store.lexical_candidates(
            conn,
            query.lexical_text,
            n,
            rank=settings.lexical_rank,
            normalization=settings.lexical_normalization,
            skill_id=skill_id,
            doc_type=doc_type,
        )
        if branches != "dense"
        else []
    )

    top_cosine = dense[0][1] if dense else None
    if apply_threshold and (top_cosine is None or top_cosine < settings.tau):
        return SearchResult(query, [], top_cosine, settings.tau, refused=True)

    rankings: list[list[str]] = []
    if branches != "lexical":
        rankings.append([chunk_id for chunk_id, _ in dense])
    if branches != "dense":
        rankings.append([chunk_id for chunk_id, _ in lexical])
    fused = rrf_fuse(rankings, settings.rrf_k)[:k]

    rows = store.get_hits(conn, [chunk_id for chunk_id, _ in fused])
    dense_by_id = {cid: (rank, cos) for rank, (cid, cos) in enumerate(dense, start=1)}
    lexical_by_id = {cid: (rank, score) for rank, (cid, score) in enumerate(lexical, start=1)}
    hits: list[Hit] = []
    for chunk_id, rrf_score in fused:
        row = rows[chunk_id]
        dense_rank, dense_cosine = (
            dense_by_id.get(chunk_id, (None, None)) if branches != "lexical" else (None, None)
        )
        lexical_rank, lexical_score = lexical_by_id.get(chunk_id, (None, None))
        hits.append(
            Hit(
                chunk_id=chunk_id,
                text=row.text,
                section_title=row.section_title,
                doc_type=row.doc_type,
                skills=row.skills,
                source_title=row.source_title,
                license=row.license,
                attribution=row.attribution,
                rrf_score=rrf_score,
                dense_rank=dense_rank,
                dense_cosine=dense_cosine,
                lexical_rank=lexical_rank,
                lexical_score=lexical_score,
            )
        )
    return SearchResult(query, hits, top_cosine, settings.tau, refused=False)
