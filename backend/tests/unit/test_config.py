from pathlib import Path

import pytest

from learning_agent.config import DEFAULT_CONFIG, ConfigError, load_settings


def test_default_config_is_valid() -> None:
    settings = load_settings(DEFAULT_CONFIG)
    assert settings.embedding.model_id == "BAAI/bge-m3"
    assert settings.embedding.dim == 1024
    assert settings.retrieval.dense_query == "b"  # R5-D2
    assert settings.retrieval.branches == "hybrid"
    assert settings.retrieval.k <= settings.retrieval.candidates_per_branch


def test_env_var_selects_another_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other.yaml"
    other.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace("batch_size: 16", "batch_size: 4"),
        encoding="utf-8",
    )
    monkeypatch.setenv("LEARNING_AGENT_CONFIG", str(other))
    assert load_settings().embedding.batch_size == 4


def test_hash_ignores_comments_but_not_values(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    commented = tmp_path / "commented.yaml"
    commented.write_text("# another comment\n" + text, encoding="utf-8")
    changed = tmp_path / "changed.yaml"
    changed.write_text(text.replace("batch_size: 16", "batch_size: 8"), encoding="utf-8")
    base = load_settings(DEFAULT_CONFIG).config_hash()
    assert load_settings(commented).config_hash() == base
    assert load_settings(changed).config_hash() != base


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("device: auto", "device: gpu", "device"),
        ("revision: 5617a9f61b028005a4858fdac845db406aefb181", "revision: main", "revision"),
        ("dim: 1024", "dim: 0", "dim"),
        ("fp16: true", "fp16: true\n  pooling: cls", "pooling"),
        ("dense_query: b", "dense_query: d", "dense_query"),
        ("branches: hybrid", "branches: both", "branches"),
        ("lexical_rank: ts_rank", "lexical_rank: bm25", "lexical_rank"),
        ("lexical_normalization: 1", "lexical_normalization: 64", "lexical_normalization"),
        ("tau: 0.5", "tau: 1.5", "tau"),
        ("max_query_chars: 1000", "max_query_chars: 0", "max_query_chars"),
        ("\n  k: 5", "\n  k: 21", "candidates_per_branch"),
        ("rrf_k: 60", "rrf_k: 60\n  rerank: true", "rerank"),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_settings(path)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing.yaml"):
        load_settings(tmp_path / "missing.yaml")
