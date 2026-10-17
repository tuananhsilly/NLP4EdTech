"""Render CNXML content (paragraphs, lists, tables, figures, inline math) as plain text lines.

ElementTree stores the text that follows a child in `child.tail`, not in the parent, so every
renderer here walks `e.text`, then each child followed by its `tail`. Inline content is joined
into one line with collapsed whitespace; block children (lists, tables, figures, line breaks)
start new lines. Unknown elements raise UnsupportedCNXML.

Images (R2-D2): `alt` texts of images inside step tables were found mathematically wrong, so
they are dropped and the table's `aria-label` is used instead; other images become
"[Figure: <alt>]".
"""

import xml.etree.ElementTree as ET

from learning_agent.kb.mathml import MATHML, math_to_text

CNXML = "{http://cnx.rice.edu/cnxml}"

_INLINE = {"emphasis", "term", "span", "link", "sub", "sup", "quote", "foreign", "label"}
_BLOCK = {"para", "note", "equation", "figure", "caption", "meaning", "item", "title"}
_SKIP = {"image", "colspec", "metadata"}


class UnsupportedCNXML(ValueError):
    """A CNXML element this renderer does not know how to handle."""


def name(e: ET.Element) -> str:
    return e.tag.removeprefix(CNXML)


def element_id(e: ET.Element) -> str:
    value = e.get("id")
    if not value:
        raise UnsupportedCNXML(f"<{name(e)}> has no id")
    return value


class _Lines:
    """Accumulates inline pieces into the current line; `newline()` closes it."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._current: list[str] = []

    def add(self, text: str | None) -> None:
        if text:
            self._current.append(text)

    def newline(self) -> None:
        line = " ".join("".join(self._current).split())
        if line:
            self.lines.append(line)
        self._current = []

    def extend(self, lines: list[str]) -> None:
        self.newline()
        self.lines.extend(lines)


def lines(e: ET.Element, *, in_table: bool = False) -> list[str]:
    """Plain-text lines for the content of `e` (its own text, children and their tails)."""
    out = _Lines()
    out.add(e.text)
    for child in e:
        _render_child(child, out, in_table=in_table)
        out.add(child.tail)
    out.newline()
    return out.lines


def block_lines(e: ET.Element) -> list[str]:
    """Plain-text lines for `e` itself, e.g. a <table> or <list> standing as its own block.

    `lines(table)` would render the table's children (<tgroup>), not the table.
    """
    out = _Lines()
    _render_child(e, out, in_table=False)
    out.newline()
    return out.lines


def inline(e: ET.Element) -> str:
    """The content of `e` as a single line."""
    return " ".join(lines(e))


def _render_child(child: ET.Element, out: _Lines, *, in_table: bool) -> None:
    if child.tag == MATHML + "math":
        out.add(math_to_text(child))
        return
    if not child.tag.startswith(CNXML):
        raise UnsupportedCNXML(f"unexpected element {child.tag}")
    n = name(child)
    if n in _SKIP:
        return
    if n in _INLINE:
        out.add(inline(child))
    elif n == "newline":
        out.newline()
    elif n == "media":
        if not in_table:  # R2-D2: alt texts inside step tables are not trusted
            out.extend([f"[Figure: {child.get('alt', '').strip()}]"])
    elif n == "list":
        out.extend([f"- {inline(item)}" for item in child if name(item) == "item"])
    elif n == "table":
        out.extend(table_lines(child))
    elif n in _BLOCK:
        out.extend(lines(child, in_table=in_table))
    else:
        raise UnsupportedCNXML(f"unsupported CNXML element <{n}>")


def table_lines(table: ET.Element) -> list[str]:
    """One line per row. Two-column step tables read "<explanation>: <math>".

    If any cell holds an image, the steps shown only as images are lost, so the table's
    aria-label (or summary) is appended as a description.
    """
    out: list[str] = []
    for row in table.iter(CNXML + "row"):
        cells = [inline_in_table(entry) for entry in row if name(entry) == "entry"]
        cells = [c for c in cells if c]
        if len(cells) == 2:
            out.append(f"{cells[0].rstrip('.')}: {cells[1]}")
        elif cells:
            out.append(" | ".join(cells))
    has_image = any(True for _ in table.iter(CNXML + "media"))
    description = table.get("aria-label") or table.get("summary")
    if has_image and description:
        out.append(f"(Steps: {description.strip()})")
    return out


def inline_in_table(e: ET.Element) -> str:
    return " ".join(lines(e, in_table=True))
