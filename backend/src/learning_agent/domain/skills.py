"""Skill graph of the fractions strand: skills, prerequisites and a topological order (07 §6.2).

The graph is a DAG (directed acyclic graph): an edge `p -> s` means "p is a prerequisite of s".
It is validated once on construction, so every `SkillGraph` instance is known to be acyclic.
"""

import heapq
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from learning_agent.domain._loading import CurriculumError, load_yaml_model

# "frac" followed by one or more lowercase dot-separated segments, e.g. "frac.add_sub.like_den".
SkillId = Annotated[str, Field(pattern=r"^frac(\.[a-z][a-z_]*)+$")]


class Skill(BaseModel):
    """One skill of the fractions strand, named after sections of Prealgebra 2e (ADR-15).

    There is deliberately no grade or external curriculum code (ADR-15); grade scope is
    controlled by which book sections are ingested (data/source_manifest.yaml).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: SkillId
    name_vi: str = Field(min_length=1)
    name_en: str = Field(min_length=1)
    prerequisites: tuple[SkillId, ...] = ()


class _SkillsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[Skill]


class SkillGraph:
    """Validated, read-only skill graph. Iteration order follows the input order."""

    def __init__(self, skills: Iterable[Skill]) -> None:
        self._skills: dict[str, Skill] = {}
        for s in skills:
            if s.id in self._skills:
                raise CurriculumError(f"duplicate skill id: {s.id}")
            self._skills[s.id] = s
        for s in self._skills.values():
            seen: set[str] = set()
            for p in s.prerequisites:
                if p not in self._skills:
                    raise CurriculumError(f"{s.id} requires unknown skill: {p}")
                if p in seen:
                    raise CurriculumError(f"{s.id} lists prerequisite {p} twice")
                seen.add(p)
        self._order = _topological_order(self._skills)

    def __contains__(self, skill_id: object) -> bool:
        return skill_id in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def ids(self) -> list[str]:
        return list(self._skills)

    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def prerequisites(self, skill_id: str) -> tuple[str, ...]:
        return self._skills[skill_id].prerequisites

    def topo_order(self) -> list[str]:
        """Every skill appears after all of its prerequisites; ties keep the input order."""
        return list(self._order)


def load_skills(path: Path) -> SkillGraph:
    return SkillGraph(load_yaml_model(path, _SkillsFile).skills)


def _topological_order(skills: dict[str, Skill]) -> list[str]:
    """Kahn's algorithm with a priority queue keyed by input position (deterministic output).

    Raises CurriculumError naming one concrete cycle if the graph is not acyclic.
    """
    position = {skill_id: i for i, skill_id in enumerate(skills)}
    # in_degree[s] = number of prerequisites of s not yet placed in the order
    in_degree = {skill_id: len(s.prerequisites) for skill_id, s in skills.items()}
    # dependents[p] = skills that list p as a prerequisite (edges reversed for fast lookup)
    dependents: dict[str, list[str]] = {skill_id: [] for skill_id in skills}
    for skill_id, s in skills.items():
        for p in s.prerequisites:
            dependents[p].append(skill_id)

    ready = [(position[skill_id], skill_id) for skill_id, d in in_degree.items() if d == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, skill_id = heapq.heappop(ready)
        order.append(skill_id)
        for dependent in dependents[skill_id]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heapq.heappush(ready, (position[dependent], dependent))

    if len(order) < len(skills):
        stuck = [skill_id for skill_id in skills if in_degree[skill_id] > 0]
        cycle = _find_cycle(stuck, skills)
        raise CurriculumError(
            "prerequisite cycle (each skill requires the next): " + " -> ".join(cycle)
        )
    return order


def _find_cycle(stuck: list[str], skills: dict[str, Skill]) -> list[str]:
    """Return one cycle among the skills Kahn's algorithm could not place, closed (a -> ... -> a).

    Every stuck skill has at least one stuck prerequisite (otherwise its in-degree would have
    reached 0), so following stuck prerequisites must eventually revisit a skill. The part of
    the walk from that skill onwards is a cycle; skills before it are only downstream of it.
    """
    stuck_set = set(stuck)
    index_in_walk: dict[str, int] = {}
    walk: list[str] = []
    current = stuck[0]
    while current not in index_in_walk:
        index_in_walk[current] = len(walk)
        walk.append(current)
        current = next(p for p in skills[current].prerequisites if p in stuck_set)
    return [*walk[index_in_walk[current] :], current]
