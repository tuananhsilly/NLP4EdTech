"""Typed runtime settings, read from backend/config/default.yaml (or LEARNING_AGENT_CONFIG)."""

import hashlib
import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class ConfigError(ValueError):
    """The config file is missing, not YAML, or does not match the settings schema."""


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dim: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: Literal["auto", "cuda", "cpu"]
    fp16: bool
    include_section_title: bool


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(gt=0)
    candidates_per_branch: int = Field(gt=0)
    rrf_k: int = Field(ge=0)
    tau: float = Field(ge=-1.0, le=1.0)  # on the cosine similarity of the best dense hit
    max_query_chars: int = Field(gt=0)
    dense_query: Literal["a", "b", "c"]
    branches: Literal["hybrid", "dense", "lexical"]
    lexical_rank: Literal["ts_rank", "ts_rank_cd"]
    lexical_normalization: int = Field(ge=0, le=63)  # Postgres bit mask: 1|2|4|8|16|32

    @model_validator(mode="after")
    def _k_fits_candidates(self) -> Self:
        if self.k > self.candidates_per_branch:
            raise ValueError(f"k ({self.k}) exceeds candidates_per_branch")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding: EmbeddingSettings
    retrieval: RetrievalSettings

    def config_hash(self) -> str:
        """Hash of the parsed values (not of the file text), so comments do not change it."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


def load_settings(path: Path | None = None) -> Settings:
    path = path or Path(os.environ.get("LEARNING_AGENT_CONFIG") or DEFAULT_CONFIG)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Settings.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
