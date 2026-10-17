"""Domain knowledge of the fractions strand: skill graph and EN-VI glossary (07 §5.1).

Deterministic code only; must not import `learning_agent.llm` (07 §5.2, rule 1).
"""

from learning_agent.domain._loading import CurriculumError
from learning_agent.domain.glossary import Glossary, GlossaryTerm, load_glossary
from learning_agent.domain.skills import Skill, SkillGraph, SkillId, load_skills

__all__ = [
    "CurriculumError",
    "Glossary",
    "GlossaryTerm",
    "Skill",
    "SkillGraph",
    "SkillId",
    "load_glossary",
    "load_skills",
]
