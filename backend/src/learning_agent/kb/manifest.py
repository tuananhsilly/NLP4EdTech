"""Typed view of data/source_manifest.yaml (R0): modules to read and sections to skip."""

import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModuleEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^m\d+$")
    section: str
    title: str
    scope: Literal["in", "partial", "out"]
    exclude_sections: tuple[str, ...] = ()
    note: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    publisher: str
    edition: str
    language: str
    collection_id: str
    book_url: str
    license: str
    license_url: str
    attribution: str
    repo_url: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    retrieved_at: datetime.date
    format: str
    files: dict[str, str]
    modules: tuple[ModuleEntry, ...]


class SourceManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: tuple[SourceEntry, ...]


def load_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
