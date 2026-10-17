from pathlib import Path

import pytest

from learning_agent.domain import CurriculumError, Skill, SkillGraph, load_skills

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS_FILE = REPO_ROOT / "data" / "curriculum" / "skills.yaml"


def skill(skill_id: str, *prerequisites: str) -> Skill:
    return Skill(id=skill_id, name_vi=skill_id, name_en=skill_id, prerequisites=prerequisites)


# --- topological order -------------------------------------------------------------------


def test_topo_order_puts_prerequisites_first() -> None:
    graph = SkillGraph([skill("frac.c", "frac.b"), skill("frac.b", "frac.a"), skill("frac.a")])
    assert graph.topo_order() == ["frac.a", "frac.b", "frac.c"]


def test_topo_order_breaks_ties_by_file_order() -> None:
    # frac.y and frac.x both only need frac.root: file order (y before x) decides.
    graph = SkillGraph(
        [skill("frac.root"), skill("frac.y", "frac.root"), skill("frac.x", "frac.root")]
    )
    assert graph.topo_order() == ["frac.root", "frac.y", "frac.x"]


def test_topo_order_handles_diamond() -> None:
    #   a -> b -> d
    #   a -> c -> d      (d needs both b and c)
    graph = SkillGraph(
        [
            skill("frac.d", "frac.b", "frac.c"),
            skill("frac.c", "frac.a"),
            skill("frac.b", "frac.a"),
            skill("frac.a"),
        ]
    )
    order = graph.topo_order()
    assert order.index("frac.a") < order.index("frac.b") < order.index("frac.d")
    assert order.index("frac.a") < order.index("frac.c") < order.index("frac.d")


# --- invalid graphs ----------------------------------------------------------------------


def test_duplicate_id_is_rejected() -> None:
    with pytest.raises(CurriculumError, match="duplicate skill id: frac.a"):
        SkillGraph([skill("frac.a"), skill("frac.a")])


def test_unknown_prerequisite_is_rejected() -> None:
    with pytest.raises(CurriculumError, match="frac.b requires unknown skill: frac.missing"):
        SkillGraph([skill("frac.a"), skill("frac.b", "frac.missing")])


def test_duplicate_prerequisite_is_rejected() -> None:
    with pytest.raises(CurriculumError, match="frac.b lists prerequisite frac.a twice"):
        SkillGraph([skill("frac.a"), skill("frac.b", "frac.a", "frac.a")])


@pytest.mark.parametrize(
    ("skills", "cycle"),
    [
        # self-loop
        ([skill("frac.a", "frac.a")], "frac.a -> frac.a"),
        # two-node cycle
        ([skill("frac.a", "frac.b"), skill("frac.b", "frac.a")], "frac.a -> frac.b -> frac.a"),
        # three-node cycle hanging off a valid root; frac.root itself is fine
        (
            [
                skill("frac.root"),
                skill("frac.a", "frac.root", "frac.c"),
                skill("frac.b", "frac.a"),
                skill("frac.c", "frac.b"),
            ],
            "frac.a -> frac.c -> frac.b -> frac.a",
        ),
    ],
)
def test_cycle_is_rejected_and_reported(skills: list[Skill], cycle: str) -> None:
    with pytest.raises(CurriculumError, match="prerequisite cycle") as excinfo:
        SkillGraph(skills)
    assert cycle in str(excinfo.value)


def test_cycle_report_excludes_skills_only_downstream_of_cycle() -> None:
    # frac.after depends on the cycle but is not part of it.
    skills = [skill("frac.a", "frac.b"), skill("frac.b", "frac.a"), skill("frac.after", "frac.a")]
    with pytest.raises(CurriculumError) as excinfo:
        SkillGraph(skills)
    assert "frac.after" not in str(excinfo.value)


# --- model validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id", ["concept", "frac", "frac.", "frac.Add", "frac.add-sub", "math.x"]
)
def test_invalid_skill_id_is_rejected(bad_id: str) -> None:
    with pytest.raises(ValueError):
        skill(bad_id)


@pytest.mark.parametrize("field", ["grade", "gdpt_ref", "ccss_ref"])
def test_fields_dropped_by_adr15_are_rejected(tmp_path: Path, field: str) -> None:
    # ADR-15: skills carry no grade or external curriculum reference; extra="forbid" keeps
    # them from silently coming back through the YAML file.
    path = tmp_path / "skills.yaml"
    path.write_text(
        f"skills:\n  - {{id: frac.a, name_vi: A, name_en: A, {field}: x}}\n", encoding="utf-8"
    )
    with pytest.raises(CurriculumError, match=field):
        load_skills(path)


# --- loading from YAML -------------------------------------------------------------------


def test_load_skills_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "skills.yaml"
    path.write_text(
        "skills:\n"
        "  - {id: frac.a, name_vi: A, name_en: A}\n"
        "  - {id: frac.b, name_vi: B, name_en: B Skill, prerequisites: [frac.a]}\n",
        encoding="utf-8",
    )
    graph = load_skills(path)
    assert graph.prerequisites("frac.b") == ("frac.a",)
    assert graph.get("frac.b").name_en == "B Skill"
    assert "frac.a" in graph
    assert len(graph) == 2


def test_load_skills_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "skills.yaml"
    path.write_text(
        "skills:\n  - {id: frac.a, name_vi: A, name_en: A, level: 2}\n", encoding="utf-8"
    )
    with pytest.raises(CurriculumError, match="level"):
        load_skills(path)


def test_load_skills_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "skills.yaml"
    path.write_text("skills: [unclosed\n", encoding="utf-8")
    with pytest.raises(CurriculumError, match="invalid YAML"):
        load_skills(path)


# --- the real data file ------------------------------------------------------------------


def test_real_skill_graph_is_valid() -> None:
    graph = load_skills(SKILLS_FILE)
    order = graph.topo_order()
    assert order[0] == "frac.concept"
    assert sorted(order) == sorted(graph.ids())
    for skill_id in order:
        for prerequisite in graph.prerequisites(skill_id):
            assert order.index(prerequisite) < order.index(skill_id)
