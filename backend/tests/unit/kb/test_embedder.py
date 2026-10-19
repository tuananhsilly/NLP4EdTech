import numpy as np
import pytest

from learning_agent.config import DEFAULT_CONFIG, load_settings
from learning_agent.kb.embedder import EmbeddingError, HashEmbedder, resolve_device


@pytest.mark.parametrize(
    ("preference", "cuda", "device"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cuda", True, "cuda"), ("cpu", True, "cpu")],
)
def test_resolve_device(preference: str, cuda: bool, device: str) -> None:
    assert resolve_device(preference, cuda) == device  # type: ignore[arg-type]


def test_cuda_required_but_missing_raises() -> None:
    with pytest.raises(EmbeddingError, match="no CUDA device"):
        resolve_device("cuda", cuda_available=False)


def test_hash_embedder_gives_unit_vectors_of_the_right_shape() -> None:
    e = HashEmbedder(dim=8)
    docs = e.embed_documents(["a", "b", "c"])
    assert docs.shape == (3, 8)
    assert docs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-6)
    assert e.embed_documents([]).shape == (0, 8)


def test_hash_embedder_is_deterministic_and_query_matches_document() -> None:
    e = HashEmbedder(dim=8)
    np.testing.assert_array_equal(e.embed_query("x"), HashEmbedder(dim=8).embed_documents(["x"])[0])
    assert not np.array_equal(e.embed_query("x"), e.embed_query("y"))


# --- the real model (skipped unless it is already in the Hugging Face cache) --------------


def _model_is_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    s = load_settings(DEFAULT_CONFIG).embedding
    return isinstance(try_to_load_from_cache(s.model_id, "config.json", revision=s.revision), str)


@pytest.mark.skipif(not _model_is_cached(), reason="model not downloaded (run kb.ingest once)")
def test_real_model_matches_the_config_and_is_cross_lingual() -> None:
    from learning_agent.kb.embedder import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder(load_settings(DEFAULT_CONFIG).embedding)
    docs = e.embed_documents(
        [
            "reciprocal: The reciprocal of the fraction a/b is b/a.",
            "Convert a fraction to a decimal: divide the numerator by the denominator.",
        ]
    )
    assert docs.shape == (2, e.dim)
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-3)
    english = docs @ e.embed_query("What is the reciprocal of a fraction?")
    vietnamese = docs @ e.embed_query("Làm sao đổi phân số sang số thập phân?")
    assert english[0] > english[1]
    assert vietnamese[1] > vietnamese[0]  # cross-lingual: a Vietnamese question, English text
    # Not asserted: "Số nghịch đảo ... là gì?" scores the decimal text higher (0.411 vs 0.373),
    # the model does not map this term; R5 maps it to "reciprocal" with the glossary first.
    assert e.count_tokens("7/12") >= 3  # special tokens included
