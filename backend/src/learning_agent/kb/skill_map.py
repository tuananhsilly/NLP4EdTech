"""Which skills each in-scope chunk teaches (data/section_skills.yaml, R3-D2).

The map is checked both ways against the chunks: an in-scope chunk without a skill and an entry
that matches no in-scope chunk are both errors, so a change to the manifest or the chunker that
moves chunks between sections cannot silently drop them from skill-filtered retrieval.
"""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from learning_agent.domain import CurriculumError, SkillGraph, SkillId
from learning_agent.domain._loading import load_yaml_model
from learning_agent.kb.chunk import REPO_ROOT
from learning_agent.kb.chunker import Chunk

DEFAULT_SKILL_MAP = REPO_ROOT / "data" / "section_skills.yaml"
DEFAULT_SKILLS = REPO_ROOT / "data" / "curriculum" / "skills.yaml"
DEFAULT_GLOSSARY = REPO_ROOT / "data" / "curriculum" / "glossary.yaml"
Skills = Annotated[tuple[SkillId, ...], Field(min_length=1)]


class SectionEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str = Field(pattern=r"^m\d+$")
    section: str
    skills: Skills


class SkillMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sections: tuple[SectionEntry, ...]
    chunks: dict[str, Skills] = {}

    @model_validator(mode="after")
    def _no_duplicates(self) -> "SkillMap":
        keys = [(e.module, e.section) for e in self.sections]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"sections listed twice: {duplicates}")
        owners = [(f"{e.module} / {e.section}", e.skills) for e in self.sections]
        for owner, skills in [*owners, *self.chunks.items()]:
            if len(set(skills)) != len(skills):
                raise ValueError(f"{owner} lists a skill twice")
        return self

    def all_skill_ids(self) -> set[str]:
        return {s for e in self.sections for s in e.skills} | {
            s for skills in self.chunks.values() for s in skills
        }


def load_skill_map(path: Path, skills: SkillGraph) -> SkillMap:
    skill_map = load_yaml_model(path, SkillMap)
    unknown = sorted(s for s in skill_map.all_skill_ids() if s not in skills)
    if unknown:
        raise CurriculumError(f"{path}: unknown skills {unknown}")
    return skill_map


def assign_skills(chunks: list[Chunk], skill_map: SkillMap) -> dict[str, Skills]:
    """Skills of every in-scope chunk, by chunk id (out-of-scope chunks are left out).

    A `chunks:` entry wins over the chunk's section entry. Raises CurriculumError listing all
    in-scope chunks without a skill and all entries that match no in-scope chunk.
    """
    by_section = {(e.module, e.section): e.skills for e in skill_map.sections}
    in_scope = [c for c in chunks if c.in_scope]
    assigned: dict[str, Skills] = {}
    used_sections: set[tuple[str, str]] = set()
    for c in in_scope:
        if c.id in skill_map.chunks:
            assigned[c.id] = skill_map.chunks[c.id]
        elif (key := (c.module_id, c.section_title)) in by_section:
            assigned[c.id] = by_section[key]
            used_sections.add(key)

    problems: list[str] = []
    unmapped = [f"{c.id} ({c.section_title})" for c in in_scope if c.id not in assigned]
    if unmapped:
        problems.append(f"in-scope chunks without a skill: {unmapped}")
    unused = [f"{m} / {s}" for m, s in sorted(set(by_section) - used_sections)]
    unused += sorted(set(skill_map.chunks) - {c.id for c in in_scope})
    if unused:
        problems.append(f"entries matching no in-scope chunk: {unused}")
    if problems:
        raise CurriculumError("; ".join(problems))
    return assigned
