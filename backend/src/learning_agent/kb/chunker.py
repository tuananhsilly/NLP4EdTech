"""Structural chunking of one CNXML module (R2 of M6a, decisions R2-D1..D5 and G1).

One chunk per meaning unit chosen by the book's own structure:
  worked_example  an <example> (problem + solution)
  procedure       a <note class="howto">
  key_concepts    the "Key Concepts" section
  definition      a glossary <definition>
  theory          consecutive paragraphs, plain notes, lists and figures of a topic section,
                  grouped up to `max_words` and split only between blocks
Try It notes, "Be Prepared" notes, media links, manipulative activities and the exercise
sections are not knowledge-base content (R2-D1) and are skipped.

Chunks out of scope are kept and flagged (R2-D3), so the report can show what was dropped and
why: the module is `out`, the section is excluded in the manifest, or the chunk uses negative
numbers, or is a worked example with variables (G1: positive fractions only; RV1).
Chunk ids are "<module id>:<element id>" (R2-D5), stable for the pinned source commit.
"""

import re
import xml.etree.ElementTree as ET
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from learning_agent.kb.cnxml import (
    CNXML,
    UnsupportedCNXML,
    block_lines,
    element_id,
    inline,
    lines,
    name,
)
from learning_agent.kb.manifest import ModuleEntry
from learning_agent.kb.mathml import MATHML, has_negative, has_variable

__all__ = ["Chunk", "DocType", "ScopeReason", "UnsupportedCNXML", "chunk_module"]

DocType = Literal["theory", "worked_example", "procedure", "key_concepts", "definition"]
ScopeReason = Literal["module_out", "excluded_section", "negative_numbers", "variables"]

_SKIPPED_SECTION_CLASSES = {
    "section-exercises",
    "practice-perfect",
    "everyday",
    "writing",
    "self-check",
}
_SKIPPED_NOTE_CLASSES = {"try", "be-prepared", "media-2", "manipulative-math"}
_THEORY_BLOCKS = {"para", "list", "equation", "figure", "table", "media", "note"}
_NEGATIVE_WORDS = re.compile(r"\b(negative|opposite)s?\b", re.IGNORECASE)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^m\d+:[\w.-]+$")
    module_id: str
    section_title: str
    doc_type: DocType
    text: str = Field(min_length=1)
    in_scope: bool
    scope_reason: ScopeReason | None

    @model_validator(mode="after")
    def _scope_reason_matches_flag(self) -> "Chunk":
        if self.in_scope != (self.scope_reason is None):
            raise ValueError("scope_reason must be set exactly when in_scope is false")
        return self


# A theory block waiting to be grouped: (element id, its lines, the element).
_TheoryItem = tuple[str, list[str], ET.Element]


class _ModuleChunker:
    def __init__(self, module: ModuleEntry, max_words: int) -> None:
        self.module = module
        self.max_words = max_words
        self.chunks: list[Chunk] = []

    # --- emitting ------------------------------------------------------------------------

    def emit(
        self,
        source_id: str,
        doc_type: DocType,
        text_lines: list[str],
        elements: list[ET.Element],
        section_title: str,
        excluded: bool,
    ) -> None:
        text = "\n".join(text_lines).strip()
        if not text:
            return
        reason = self._scope_reason(doc_type, text, elements, excluded)
        self.chunks.append(
            Chunk(
                id=f"{self.module.id}:{source_id}",
                module_id=self.module.id,
                section_title=section_title,
                doc_type=doc_type,
                text=text,
                in_scope=reason is None,
                scope_reason=reason,
            )
        )

    def _scope_reason(
        self, doc_type: DocType, text: str, elements: list[ET.Element], excluded: bool
    ) -> ScopeReason | None:
        if self.module.scope == "out":
            return "module_out"
        if excluded:
            return "excluded_section"
        maths = [m for e in elements for m in e.iter(MATHML + "math")]
        if any(has_negative(m) for m in maths) or _NEGATIVE_WORDS.search(text):
            return "negative_numbers"
        # RV1: only worked examples are algebra when they use letters; definitions, rules and
        # explanations use letters as general notation (a/c + b/c = (a + b)/c) and stay in scope.
        if doc_type == "worked_example" and any(has_variable(m) for m in maths):
            return "variables"
        return None

    def flush(self, theory: list[_TheoryItem], section_title: str, excluded: bool) -> None:
        """Group pending theory blocks into chunks of at most `max_words` (split between blocks)."""
        group: list[_TheoryItem] = []
        words = 0
        for item in theory:
            item_words = sum(len(line.split()) for line in item[1])
            if group and words + item_words > self.max_words:
                self._emit_theory(group, section_title, excluded)
                group, words = [], 0
            group.append(item)
            words += item_words
        if group:
            self._emit_theory(group, section_title, excluded)
        theory.clear()

    def _emit_theory(self, group: list[_TheoryItem], section_title: str, excluded: bool) -> None:
        text_lines = [line for _, item_lines, _ in group for line in item_lines]
        elements = [e for _, _, e in group]
        self.emit(group[0][0], "theory", text_lines, elements, section_title, excluded)

    # --- walking the structure -----------------------------------------------------------

    def walk(self, container: ET.Element, section_title: str, excluded: bool) -> None:
        theory: list[_TheoryItem] = []
        for child in container:
            n = name(child)
            if n == "title" or child.tag.startswith(MATHML):
                continue
            if n == "section":
                self.flush(theory, section_title, excluded)
                self.section(child, excluded)
            elif n == "example":
                self.flush(theory, section_title, excluded)
                self.example(child, section_title, excluded)
            elif n == "note" and child.get("class") == "howto":
                self.flush(theory, section_title, excluded)
                self.emit(
                    element_id(child), "procedure", lines(child), [child], section_title, excluded
                )
            elif n == "note" and child.get("class") in _SKIPPED_NOTE_CLASSES:
                continue
            elif n == "exercise" or (n == "figure" and child.get("class") == "splash"):
                continue  # stray exercises and decorative chapter photos are not KB content
            elif n in _THEORY_BLOCKS:
                theory.append((element_id(child), block_lines(child), child))
            else:
                raise UnsupportedCNXML(f"unsupported CNXML element <{n}> in a section")
        self.flush(theory, section_title, excluded)

    def section(self, section: ET.Element, parent_excluded: bool) -> None:
        classes = set((section.get("class") or "").split())
        if classes & _SKIPPED_SECTION_CLASSES:
            return
        title_element = section.find(CNXML + "title")
        title = inline(title_element) if title_element is not None else ""
        excluded = parent_excluded or title in self.module.exclude_sections
        if "key-concepts" in classes:
            body = [line for c in section if name(c) != "title" for line in block_lines(c)]
            self.emit(element_id(section), "key_concepts", body, [section], title, excluded)
        else:
            self.walk(section, title, excluded)

    def example(self, example: ET.Element, section_title: str, excluded: bool) -> None:
        text_lines: list[str] = []
        for exercise in example.iter(CNXML + "exercise"):
            for part in exercise:
                label = {"problem": "Example", "solution": "Solution"}.get(name(part))
                if label is None:
                    raise UnsupportedCNXML(f"unsupported <{name(part)}> in an example")
                body = [line for c in part if name(c) != "title" for line in block_lines(c)]
                if body:
                    text_lines.append(f"{label}: {body[0]}")
                    text_lines.extend(body[1:])
        self.emit(
            element_id(example), "worked_example", text_lines, [example], section_title, excluded
        )

    def glossary(self, glossary: ET.Element) -> None:
        for definition in glossary.iter(CNXML + "definition"):
            term = definition.find(CNXML + "term")
            meaning = definition.find(CNXML + "meaning")
            if term is None or meaning is None:
                raise UnsupportedCNXML("<definition> needs <term> and <meaning>")
            text = f"{inline(term)}: {inline(meaning)}"
            self.emit(element_id(definition), "definition", [text], [definition], "Glossary", False)


def chunk_module(root: ET.Element, module: ModuleEntry, *, max_words: int = 250) -> list[Chunk]:
    """Chunks of one parsed CNXML <document>, in document order."""
    chunker = _ModuleChunker(module, max_words)
    content = root.find(CNXML + "content")
    if content is None:
        raise UnsupportedCNXML(f"{module.id}: document has no <content>")
    chunker.walk(content, module.title, excluded=False)
    glossary = root.find(CNXML + "glossary")
    if glossary is not None:
        chunker.glossary(glossary)
    return chunker.chunks
