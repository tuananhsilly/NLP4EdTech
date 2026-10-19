"""Build the knowledge base in one command: chunk, tag skills, embed, store, report (M6a R4).

    docker compose up -d                                         # from the repo root
    uv run --env-file .env python -m learning_agent.kb.ingest    # add --probe "query" to try it

Everything that can fail slowly or for a reason outside the database (reading the sources, the
skill map, loading the model, embedding on the GPU) happens before the first write. The chunks,
their skills and vectors, and the ingest_run row are then written in one transaction, so the
database holds either the previous knowledge base or the new one, never a mix.
"""

import argparse
import datetime
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from learning_agent.config import Settings, load_settings
from learning_agent.db import connect
from learning_agent.db.migrate import migrate
from learning_agent.domain import load_skills
from learning_agent.kb import store
from learning_agent.kb.chunk import (
    DEFAULT_MANIFEST,
    DEFAULT_OUT,
    DEFAULT_SOURCES,
    REPO_ROOT,
    build_chunks,
    report,
    write_jsonl,
)
from learning_agent.kb.chunker import Chunk
from learning_agent.kb.embedder import Embedder, SentenceTransformerEmbedder, Vectors
from learning_agent.kb.manifest import load_manifest
from learning_agent.kb.skill_map import (
    DEFAULT_SKILL_MAP,
    DEFAULT_SKILLS,
    assign_skills,
    load_skill_map,
)


def document_text(chunk: Chunk, include_section_title: bool) -> str:
    """What gets embedded (R4-D3); the stored text stays the chunk text."""
    return f"{chunk.section_title}\n{chunk.text}" if include_section_title else chunk.text


@dataclass(frozen=True)
class EmbedResult:
    vectors: dict[str, Vectors]
    longest_tokens: int
    truncated: list[str]  # ids of chunks longer than the model's max_seq_length
    seconds: float


def embed_chunks(
    embedder: Embedder, chunks: Sequence[Chunk], *, include_section_title: bool, max_tokens: int
) -> EmbedResult:
    in_scope = [c for c in chunks if c.in_scope]
    texts = [document_text(c, include_section_title) for c in in_scope]
    tokens = [embedder.count_tokens(t) for t in texts]
    start = time.perf_counter()
    matrix = embedder.embed_documents(texts)
    return EmbedResult(
        vectors={c.id: matrix[i] for i, c in enumerate(in_scope)},
        longest_tokens=max(tokens, default=0),
        truncated=[c.id for c, n in zip(in_scope, tokens, strict=True) if n > max_tokens],
        seconds=time.perf_counter() - start,
    )


def git_commit() -> str | None:
    """HEAD of the checkout, with "-dirty" when there are uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if status.strip() else "")


def peak_gpu_memory_mb(device: str) -> float | None:
    if device != "cuda":
        return None
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk, tag, embed and store the knowledge base.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--skill-map", type=Path, default=DEFAULT_SKILL_MAP)
    parser.add_argument("--chunks-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-words", type=int, default=250, help="theory chunk size limit")
    parser.add_argument(
        "--probe", action="append", default=[], help="print the 5 nearest chunks for a query"
    )
    args = parser.parse_args(argv)

    settings: Settings = load_settings()
    started_at = datetime.datetime.now(datetime.UTC)
    manifest = load_manifest(args.manifest)
    (source,) = manifest.sources
    skills = load_skills(args.skills)
    chunks = build_chunks(manifest, args.sources, args.max_words)
    write_jsonl(chunks, args.chunks_out)
    chunk_skills = assign_skills(chunks, load_skill_map(args.skill_map, skills))
    print(report(chunks, args.max_words), end="\n\n")

    embedder = SentenceTransformerEmbedder(settings.embedding)
    result = embed_chunks(
        embedder,
        chunks,
        include_section_title=settings.embedding.include_section_title,
        max_tokens=settings.embedding.max_seq_length,
    )
    info = embedder.info
    stats: dict[str, object] = {
        "chunks_total": len(chunks),
        "chunks_stored": len(result.vectors),
        "device": info.device,
        "device_name": info.device_name,
        "dtype": info.dtype,
        "embed_seconds": round(result.seconds, 2),
        "longest_tokens": result.longest_tokens,
        "truncated": result.truncated,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(info.device),
    }

    with connect() as conn:
        for m in migrate(conn):
            print(f"applied {m.name}")
        column_dim = store.embedding_dim(conn)
        if column_dim != embedder.dim:
            raise SystemExit(
                f"chunk.embedding has {column_dim} dimensions but the model gives {embedder.dim}; "
                "a different model needs a new migration"
            )
        with conn.transaction():
            stored = store.load(conn, source, skills, chunks, chunk_skills, result.vectors)
            run_id = store.record_ingest_run(
                conn,
                started_at=started_at,
                source_ids=[source.id],
                git_commit=git_commit(),
                config=settings.model_dump(mode="json"),
                config_hash=settings.config_hash(),
                stats=stats,
            )
        self_hits = sum(
            store.nearest(conn, vector, 1)[0][0] == chunk_id
            for chunk_id, vector in result.vectors.items()
        )
        by_skill = store.count_by_skill(conn)
        probes = [(q, store.nearest(conn, embedder.embed_query(q), 5)) for q in args.probe]

    print("chunks by skill (a chunk can teach several):")
    for skill_id in skills.topo_order():
        print(f"  {skill_id:26}{by_skill.get(skill_id, 0):5}")
    print(
        f"\nembedded {stored} in-scope chunks (of {len(chunks)}) "
        f"with {info.model_id}@{info.revision[:8]} on {info.device} "
        f"({info.device_name}, {info.dtype}) in {result.seconds:.1f} s"
    )
    print(
        f"longest input {result.longest_tokens} tokens, truncated: {len(result.truncated)} "
        f"{result.truncated or ''}; peak GPU memory: {stats['peak_gpu_memory_mb']} MB"
    )
    print(f"HNSW self-retrieval: {self_hits}/{stored} chunks find themselves first")
    print(f"ingest_run {run_id} (config {settings.config_hash()[:12]})")
    for query, hits in probes:
        print(f"\nprobe: {query}")
        for chunk_id, distance in hits:
            print(f"  {distance:.3f}  {chunk_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
