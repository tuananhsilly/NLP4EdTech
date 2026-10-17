"""Chunking the real Prealgebra modules (needs scripts/fetch_sources.sh; skipped otherwise)."""

from pathlib import Path

import pytest

from learning_agent.kb.chunk import (
    DEFAULT_MANIFEST,
    DEFAULT_SOURCES,
    build_chunks,
    source_dir,
    write_jsonl,
)
from learning_agent.kb.chunker import Chunk
from learning_agent.kb.manifest import load_manifest

MANIFEST = load_manifest(DEFAULT_MANIFEST)

pytestmark = pytest.mark.skipif(
    not source_dir(MANIFEST, DEFAULT_SOURCES).exists(),
    reason="raw sources not fetched (scripts/fetch_sources.sh)",
)


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return build_chunks(MANIFEST, DEFAULT_SOURCES, max_words=250)


def test_ids_are_unique_and_belong_to_manifest_modules(chunks: list[Chunk]) -> None:
    module_ids = {m.id for m in MANIFEST.sources[0].modules}
    assert len({c.id for c in chunks}) == len(chunks)
    assert {c.module_id for c in chunks} <= module_ids
    assert all(c.id.startswith(c.module_id + ":") for c in chunks)


def test_no_xml_left_in_text(chunks: list[Chunk]) -> None:
    assert [c.id for c in chunks if "</" in c.text or "<m:" in c.text] == []


def test_build_is_deterministic(chunks: list[Chunk]) -> None:
    assert build_chunks(MANIFEST, DEFAULT_SOURCES, max_words=250) == chunks


def test_known_worked_example_is_rendered_and_in_scope(chunks: list[Chunk]) -> None:
    example = next(c for c in chunks if c.id == "m81289:fs-id1858347")
    assert example.doc_type == "worked_example"
    assert example.in_scope
    assert example.text.startswith("Example: Find the LCD for the fractions 7/12 and 5/18.")


def test_module_marked_out_is_entirely_out_of_scope(chunks: list[Chunk]) -> None:
    solve_equations = [c for c in chunks if c.module_id == "m81291"]
    assert solve_equations
    assert all(c.scope_reason == "module_out" for c in solve_equations)


def test_jsonl_round_trip(chunks: list[Chunk], tmp_path: Path) -> None:
    out = tmp_path / "chunks.jsonl"
    write_jsonl(chunks, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [Chunk.model_validate_json(line) for line in lines] == chunks
