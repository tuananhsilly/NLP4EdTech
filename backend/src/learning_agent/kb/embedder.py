"""Text embeddings behind a thin adapter (07 P6): the KB depends only on the Embedder protocol.

Vectors are float32, one row per text, normalized to length 1, so cosine distance in pgvector
(`<=>`) ranks them like the dot product. torch and sentence-transformers are imported only
when the real model is created, so code and tests that use HashEmbedder never load them.
"""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from learning_agent.config import EmbeddingSettings

Vectors = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """The model cannot be used as configured (device, dimension)."""


@dataclass(frozen=True)
class EmbedderInfo:
    """What produced the vectors; stored with every ingest run."""

    model_id: str
    revision: str
    device: str
    device_name: str
    dtype: str


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def info(self) -> EmbedderInfo: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Shape (dim,)."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens the model sees for `text`, special tokens included."""
        ...


def resolve_device(
    preference: Literal["auto", "cuda", "cpu"], cuda_available: bool
) -> Literal["cuda", "cpu"]:
    if preference == "cuda" and not cuda_available:
        raise EmbeddingError("device: cuda is configured but torch sees no CUDA device")
    if preference == "auto":
        return "cuda" if cuda_available else "cpu"
    return preference


class SentenceTransformerEmbedder:
    """A sentence-transformers model pinned to a revision (BGE-M3 by default, R4-D1)."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = resolve_device(settings.device, torch.cuda.is_available())
        half = device == "cuda" and settings.fp16
        # The pinned BGE-M3 revision has only pytorch_model.bin (loaded with torch's weights_only
        # unpickler). When it loads a .bin, transformers starts a background download of
        # model.safetensors from an open conversion pull request "for next time" (refs/pr/130:
        # 2.1 GB, tensors checked identical, not reviewed by the model's authors). The variable
        # stops that download; use_safetensors=False makes the .bin-only choice explicit.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        # dtype: load straight in half precision; converting after loading needs the fp32 copy
        # in GPU memory first (1.8 GB of the 4 GB card).
        model_kwargs: dict[str, object] = {"use_safetensors": False}
        if half:
            model_kwargs["dtype"] = torch.float16
        model = SentenceTransformer(
            settings.model_id, revision=settings.revision, device=device, model_kwargs=model_kwargs
        )
        model.max_seq_length = settings.max_seq_length
        model_dim = model.get_embedding_dimension()
        if model_dim != settings.dim:
            raise EmbeddingError(
                f"{settings.model_id} gives {model_dim} dimensions, config says {settings.dim}"
            )
        self._model = model
        self._batch_size = settings.batch_size
        self._dim = settings.dim
        self._info = EmbedderInfo(
            model_id=settings.model_id,
            revision=settings.revision,
            device=device,
            device_name=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            dtype="float16" if half else "float32",
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def info(self) -> EmbedderInfo:
        return self._info

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        # the library's overloads type **kwargs loosely; the result is an ndarray here
        vectors = self._model.encode(  # pyright: ignore[reportUnknownMemberType]
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        return self.embed_documents([text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text)["input_ids"])


class HashEmbedder:
    """Deterministic pseudo-random unit vectors keyed by the text; no meaning, for tests.

    Equal texts get equal vectors, so "a chunk's own text finds the chunk" holds; nothing else
    about similarity does.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def info(self) -> EmbedderInfo:
        return EmbedderInfo("hash", "-", "cpu", "cpu", "float32")

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        rows = [self._vector(t) for t in texts]
        return np.stack(rows) if rows else np.zeros((0, self._dim), dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> Vectors:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)
