"""Build data/build/chunks.jsonl from the sources in data/source_manifest.yaml, and report.

    uv run python -m learning_agent.kb.chunk                 # write chunks + print the report
    uv run python -m learning_agent.kb.chunk --show variables   # also list chunks out of scope
                                                                # for one reason (to review G1)

The output is derived from licensed content (CC BY-NC-SA) and can be rebuilt with one command,
so data/build/ is git-ignored, like data/sources/.
"""

import argparse
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from learning_agent.kb.chunker import Chunk, chunk_module
from learning_agent.kb.manifest import SourceManifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "source_manifest.yaml"
DEFAULT_SOURCES = REPO_ROOT / "data" / "sources"
DEFAULT_OUT = REPO_ROOT / "data" / "build" / "chunks.jsonl"

_REASONS = ("in scope", "negative_numbers", "variables", "excluded_section", "module_out")


def source_dir(manifest: SourceManifest, sources_root: Path) -> Path:
    """Local checkout of the (single) source, named like the repo (see scripts/fetch_sources.sh)."""
    (source,) = manifest.sources
    return sources_root / source.repo_url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]


def build_chunks(manifest: SourceManifest, sources_root: Path, max_words: int) -> list[Chunk]:
    """Chunks of every module in the manifest, in manifest order."""
    base = source_dir(manifest, sources_root)
    chunks: list[Chunk] = []
    for module in manifest.sources[0].modules:
        path = base / "modules" / module.id / "index.cnxml"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run scripts/fetch_sources.sh first")
        chunks.extend(chunk_module(ET.parse(path).getroot(), module, max_words=max_words))
    return chunks


def write_jsonl(chunks: list[Chunk], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def report(chunks: list[Chunk], max_words: int) -> str:
    counts = Counter((c.doc_type, c.scope_reason or "in scope") for c in chunks)
    doc_types = sorted({doc_type for doc_type, _ in counts})
    rows = [f"{'doc_type':16}" + "".join(f"{r:>18}" for r in _REASONS)]
    for t in doc_types:
        rows.append(f"{t:16}" + "".join(f"{counts[(t, r)]:18}" for r in _REASONS))
    rows.append(
        f"{'total':16}" + "".join(f"{sum(counts[(t, r)] for t in doc_types):18}" for r in _REASONS)
    )
    in_scope = [len(c.text.split()) for c in chunks if c.in_scope]
    over = sum(1 for n in in_scope if n > max_words)
    rows.append("")
    rows.append(f"chunks: {len(chunks)}, in scope: {len(in_scope)}")
    if in_scope:
        rows.append(
            f"in-scope words: median {statistics.median(in_scope):g}, max {max(in_scope)}, "
            f"over {max_words}: {over} (only theory is split)"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build chunks.jsonl from the source manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-words", type=int, default=250, help="theory chunk size limit")
    parser.add_argument("--show", choices=_REASONS[1:], help="list chunks out of scope for REASON")
    args = parser.parse_args(argv)

    chunks = build_chunks(load_manifest(args.manifest), args.sources, args.max_words)
    write_jsonl(chunks, args.out)
    print(report(chunks, args.max_words))
    print(f"\nwrote {len(chunks)} chunks to {args.out}")
    if args.show:
        print(f"\nout of scope ({args.show}):")
        for c in chunks:
            if c.scope_reason == args.show:
                print(f"  [{c.doc_type}] {c.id}: {c.text.splitlines()[0][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
