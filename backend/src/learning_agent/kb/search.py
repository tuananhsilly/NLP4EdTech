"""Try retrieval from the command line (R5-D3):

    uv run --env-file .env python -m learning_agent.kb.search "câu hỏi" ["another question" ...]
        [--k N] [--skill frac.div] [--doc-type definition]
        [--dense-query a|b|c] [--branches hybrid|dense|lexical] [--no-threshold]

For each question it prints how the question was prepared, the best dense cosine against tau,
and the fused ranking with each branch's rank and score. The database is checked before the
embedding model is loaded, so a stopped database fails at once. Several questions share one
model load.
"""

import argparse
import sys
from typing import get_args

import psycopg

from learning_agent.config import load_settings
from learning_agent.db import connect
from learning_agent.domain import load_glossary, load_skills
from learning_agent.kb import store
from learning_agent.kb.chunker import DocType
from learning_agent.kb.embedder import SentenceTransformerEmbedder
from learning_agent.kb.retrieval import Branches, SearchResult, search
from learning_agent.kb.skill_map import DEFAULT_GLOSSARY, DEFAULT_SKILLS


def _rank(rank: int | None, score: float | None, digits: int) -> str:
    return "-" if rank is None or score is None else f"{rank} {score:.{digits}f}"


def format_result(result: SearchResult, preview_chars: int = 70) -> str:
    q = result.query
    if not result.refused:
        verdict = "answer"
    elif result.top_cosine is None:
        verdict = "REFUSED: no chunk matches the filters"
    else:
        verdict = "REFUSED: below tau"
    cosine = "-" if result.top_cosine is None else f"{result.top_cosine:.3f}"
    lines = [
        f"query:      {q.original}",
        f"normalized: {q.normalized}",
        f"en_terms:   {', '.join(q.en_terms) or '-'}",
        f"forbidden:  {', '.join(q.forbidden) or '-'}",
        f"dense text: {q.dense_text}",
        f"top cosine: {cosine} (tau {result.tau:.3f}) -> {verdict}",
    ]
    if result.hits:
        lines.append(
            f"{'#':>2}  {'rrf':6}  {'dense':9}  {'lexical':9}  {'chunk':22}  {'type':14}  text"
        )
        for i, h in enumerate(result.hits, start=1):
            preview = " ".join(h.text.split())[:preview_chars]
            lines.append(
                f"{i:>2}  {h.rrf_score:.4f}  {_rank(h.dense_rank, h.dense_cosine, 3):9}  "
                f"{_rank(h.lexical_rank, h.lexical_score, 3):9}  {h.chunk_id:22}  "
                f"{h.doc_type:14}  {preview}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the knowledge base (hybrid + RRF + tau).")
    parser.add_argument(
        "questions", nargs="+", help="one or more questions (Vietnamese or English)"
    )
    parser.add_argument("--k", type=int, help="number of chunks (default: config)")
    parser.add_argument("--skill", help="only chunks teaching this skill id")
    parser.add_argument("--doc-type", choices=get_args(DocType))
    parser.add_argument("--dense-query", choices=["a", "b", "c"], help="default: config")
    parser.add_argument("--branches", choices=get_args(Branches), help="default: config")
    parser.add_argument(
        "--no-threshold", action="store_true", help="show the ranking even below tau"
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    skills = load_skills(DEFAULT_SKILLS)
    glossary = load_glossary(DEFAULT_GLOSSARY, skills)
    try:
        conn = connect()
    except psycopg.OperationalError as e:
        print(f"cannot reach Postgres ({e}); run `docker compose up -d`", file=sys.stderr)
        return 1

    with conn:
        embedder = SentenceTransformerEmbedder(settings.embedding)
        column_dim = store.embedding_dim(conn)
        if column_dim != embedder.dim:
            print(
                f"chunk.embedding has {column_dim} dimensions but the model gives {embedder.dim}; "
                "run the ingest first",
                file=sys.stderr,
            )
            return 1
        for i, question in enumerate(args.questions):
            try:
                result = search(
                    conn,
                    embedder,
                    glossary,
                    skills,
                    settings.retrieval,
                    question,
                    k=args.k,
                    skill_id=args.skill,
                    doc_type=args.doc_type,
                    dense_mode=args.dense_query,
                    branches=args.branches,
                    apply_threshold=not args.no_threshold,
                )
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            print(("\n" if i else "") + format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
