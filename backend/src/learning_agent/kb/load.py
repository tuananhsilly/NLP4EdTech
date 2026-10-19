"""Load data/build/chunks.jsonl into Postgres, with the skills of each chunk, and report.

    docker compose up -d                                        # from the repo root
    uv run python -m learning_agent.kb.chunk                    # build chunks.jsonl (R2)
    uv run --env-file .env python -m learning_agent.kb.load     # migrate + load + report

Only in-scope chunks are stored. If chunks.jsonl is older than the manifest or the section map,
the skill check fails; rebuild it with `python -m learning_agent.kb.chunk`.
"""

import argparse
import sys
from pathlib import Path

from learning_agent.db import connect
from learning_agent.db.migrate import migrate
from learning_agent.domain import load_skills
from learning_agent.kb import store
from learning_agent.kb.chunk import DEFAULT_MANIFEST, DEFAULT_OUT
from learning_agent.kb.chunker import Chunk
from learning_agent.kb.manifest import load_manifest
from learning_agent.kb.skill_map import (
    DEFAULT_SKILL_MAP,
    DEFAULT_SKILLS,
    assign_skills,
    load_skill_map,
)


def read_jsonl(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m learning_agent.kb.chunk` first")
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load in-scope chunks into Postgres.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--skill-map", type=Path, default=DEFAULT_SKILL_MAP)
    args = parser.parse_args(argv)

    (source,) = load_manifest(args.manifest).sources
    skills = load_skills(args.skills)
    chunks = read_jsonl(args.chunks)
    chunk_skills = assign_skills(chunks, load_skill_map(args.skill_map, skills))

    with connect() as conn:
        for m in migrate(conn):
            print(f"applied {m.name}")
        n = store.load(conn, source, skills, chunks, chunk_skills)
        by_doc_type = store.count_by_doc_type(conn)
        by_skill = store.count_by_skill(conn)

    print(f"loaded {n} in-scope chunks of {len(chunks)} from {args.chunks}\n")
    print("chunks by doc_type:")
    for doc_type, count in by_doc_type.items():
        print(f"  {doc_type:16}{count:5}")
    print("chunks by skill (a chunk can teach several):")
    for skill_id in skills.topo_order():
        print(f"  {skill_id:26}{by_skill.get(skill_id, 0):5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
