"""Shared YAML loading for curriculum data files (data/curriculum/*.yaml)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError


class CurriculumError(ValueError):
    """A curriculum data file is malformed or internally inconsistent."""


def load_yaml_model[M: BaseModel](path: Path, model: type[M]) -> M:
    """Parse `path` as YAML and validate it against `model`.

    `yaml.safe_load` only builds plain data (dict, list, str, ...), never Python objects.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CurriculumError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise CurriculumError(f"{path}: {exc}") from exc
