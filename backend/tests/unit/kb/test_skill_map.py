from pathlib import Path

import pytest
import yaml

from learning_agent.domain import CurriculumError, Skill, SkillGraph
from learning_agent.kb.chunker import Chunk
from learning_agent.kb.skill_map import SkillMap, assign_skills, load_skill_map

SKILLS = SkillGraph(
    [
        Skill(id="frac.concept", name_vi="Khái niệm", name_en="Concept"),
        Skill(id="frac.div", name_vi="Chia", name_en="Divide", prerequisites=("frac.concept",)),
    ]
)


def chunk(id_: str, section: str, *, in_scope: bool = True) -> Chunk:
    return Chunk(
        id=id_,
        module_id=id_.split(":")[0],
        section_title=section,
        doc_type="theory",
        text="x",
        in_scope=in_scope,
        scope_reason=None if in_scope else "excluded_section",
    )


CHUNKS = [
    chunk("m1:p1", "Meaning"),
    chunk("m1:p2", "Meaning"),
    chunk("m1:p3", "Divide"),
    chunk("m1:d1", "Glossary"),
    chunk("m1:x1", "Circles", in_scope=False),
]
MAP = SkillMap.model_validate(
    {
        "sections": [
            {"module": "m1", "section": "Meaning", "skills": ["frac.concept"]},
            {"module": "m1", "section": "Divide", "skills": ["frac.div", "frac.concept"]},
        ],
        "chunks": {"m1:d1": ["frac.div"]},
    }
)


def test_chunks_get_the_skills_of_their_section_or_their_own_entry() -> None:
    assert assign_skills(CHUNKS, MAP) == {
        "m1:p1": ("frac.concept",),
        "m1:p2": ("frac.concept",),
        "m1:p3": ("frac.div", "frac.concept"),
        "m1:d1": ("frac.div",),
    }


def test_chunk_entry_wins_over_its_section() -> None:
    m = MAP.model_copy(update={"chunks": {"m1:d1": ("frac.div",), "m1:p3": ("frac.concept",)}})
    assigned = assign_skills([*CHUNKS, chunk("m1:p4", "Divide")], m)
    assert assigned["m1:p3"] == ("frac.concept",)
    assert assigned["m1:p4"] == ("frac.div", "frac.concept")


def test_in_scope_chunk_without_skill_raises() -> None:
    with pytest.raises(CurriculumError, match=r"without a skill: \['m1:p9 \(Multiply\)'\]"):
        assign_skills([*CHUNKS, chunk("m1:p9", "Multiply")], MAP)


def test_entries_matching_no_in_scope_chunk_raise() -> None:
    # "Divide" only had m1:p3; an entry for an out-of-scope chunk is stale too
    chunks = [c for c in CHUNKS if c.id != "m1:p3"]
    m = MAP.model_copy(update={"chunks": {"m1:d1": ("frac.div",), "m1:x1": ("frac.div",)}})
    with pytest.raises(CurriculumError, match=r"no in-scope chunk: \['m1 / Divide', 'm1:x1'\]"):
        assign_skills(chunks, m)


MALFORMED: list[tuple[object, str]] = [
    (
        {"sections": [{"module": "m1", "section": "A", "skills": []}]},
        "at least 1 item",
    ),
    (
        {"sections": [{"module": "m1", "section": "A", "skills": ["frac.concept"]}] * 2},
        "listed twice",
    ),
    (
        {"sections": [], "chunks": {"m1:d1": ["frac.div", "frac.div"]}},
        "m1:d1 lists a skill twice",
    ),
    ({"sections": [{"module": "m1", "section": "A", "skills": ["fractions"]}]}, "pattern"),
]


@pytest.mark.parametrize(("data", "message"), MALFORMED)
def test_malformed_map_is_rejected(tmp_path: Path, data: object, message: str) -> None:
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CurriculumError, match=message):
        load_skill_map(path, SKILLS)


def test_unknown_skill_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "map.yaml"
    path.write_text(
        "sections:\n  - {module: m1, section: A, skills: [frac.concept, frac.mul]}\n",
        encoding="utf-8",
    )
    with pytest.raises(CurriculumError, match=r"unknown skills \['frac.mul'\]"):
        load_skill_map(path, SKILLS)
