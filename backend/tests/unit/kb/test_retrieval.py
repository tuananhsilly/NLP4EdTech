import pytest

from learning_agent.kb.retrieval import rrf_fuse


def test_rrf_sums_reciprocal_ranks_over_rankings() -> None:
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], rrf_k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_puts_a_chunk_found_by_both_branches_first() -> None:
    # "b" is second in both rankings, yet beats the two chunks ranked first by only one branch.
    assert [cid for cid, _ in rrf_fuse([["a", "b"], ["c", "b"]], rrf_k=60)] == ["b", "a", "c"]


def test_rrf_breaks_ties_by_chunk_id() -> None:
    assert [cid for cid, _ in rrf_fuse([["b"], ["a"]], rrf_k=60)] == ["a", "b"]


def test_rrf_of_a_single_ranking_keeps_its_order() -> None:
    assert [cid for cid, _ in rrf_fuse([["z", "y", "x"]], rrf_k=60)] == ["z", "y", "x"]


def test_rrf_constant_zero_gives_plain_reciprocal_rank() -> None:
    assert rrf_fuse([["a", "b"]], rrf_k=0) == [("a", 1.0), ("b", 0.5)]


def test_rrf_of_nothing_is_empty() -> None:
    assert rrf_fuse([], rrf_k=60) == []
    assert rrf_fuse([[], []], rrf_k=60) == []
