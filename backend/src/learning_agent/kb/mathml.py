"""MathML (presentation) -> one-line text, plus scope checks used by the chunker (R2, M6a).

PDF extraction breaks stacked fractions; CNXML keeps them as <m:mfrac>, so we render the tree
ourselves: 27/32 − 11/32, (2 × 5)/(3 × 5), 2 3/4, 0.3̅. Unknown elements raise
UnsupportedMathML instead of being guessed, so new notation is noticed at ingest time.
Notation follows the book (R2-D4): ÷ is kept; the multiplication dots · and ⋅ become ×.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Literal

MATHML = "{http://www.w3.org/1998/Math/MathML}"

MINUS = "−"
# Normalized operator spellings: every minus variant becomes U+2212, both dots become ×.
_OPERATOR_ALIASES = {"-": MINUS, "–": MINUS, "·": "×", "⋅": "×"}
_BINARY = {"+", MINUS, "×", "÷", "=", "≠", "<", ">", "≈", "≤", "≥"}
_SIGNS = {"+", MINUS}
_OPEN = {"(", "["}
_CLOSE = {")", "]"}
_PUNCT = {".", ",", "?", ":", ";", "!"}
_OVERLINE = "̅"  # combining overline: marks a repeating digit
_CONSTANTS = {"π"}

TokenKind = Literal["operand", "fraction", "number", "op", "open", "close", "punct", "space"]
Token = tuple[TokenKind, str]


class UnsupportedMathML(ValueError):
    """A MathML element or shape this converter does not know how to render."""


def _name(e: ET.Element) -> str:
    return e.tag.removeprefix(MATHML)


def _text(e: ET.Element) -> str:
    return " ".join((e.text or "").split())


def _operator(e: ET.Element) -> str:
    raw = _text(e)
    return _OPERATOR_ALIASES.get(raw, raw)


# --- rows: sequences of tokens -----------------------------------------------------------


def _tokens(row: ET.Element) -> list[Token]:
    """Classify the children of a row-like element (math, mrow, mstyle, mtd, ...)."""
    tokens: list[Token] = []
    for child in row:
        name = _name(child)
        if name == "mo":
            op = _operator(child)
            if op in _OPEN:
                tokens.append(("open", op))
            elif op in _CLOSE:
                tokens.append(("close", op))
            elif op in _PUNCT:
                tokens.append(("punct", op))
            else:
                tokens.append(("op", op))
        elif name == "mspace":
            tokens.append(("space", " "))
        elif name == "munder" and math_to_text(child) == "___":
            # A fill-in blank stands for a missing relation (<, >, =): treat it as an operator.
            tokens.append(("op", "___"))
        elif name == "mfrac":
            tokens.append(("fraction", math_to_text(child)))
        elif name == "mn":
            tokens.append(("number", math_to_text(child)))
        else:
            tokens.append(("operand", math_to_text(child)))
    return tokens


def _is_unary(tokens: list[Token], i: int) -> bool:
    """A sign is unary when nothing, an operator or an opening bracket comes before it."""
    kind, text = tokens[i]
    if kind != "op" or text not in _SIGNS:
        return False
    previous = [k for k, _ in tokens[:i] if k != "space"]
    return not previous or previous[-1] in ("op", "open", "punct")


def _render_row(tokens: list[Token]) -> str:
    parts: list[str] = []
    previous: TokenKind | None = None
    for i, (kind, text) in enumerate(tokens):
        if kind == "space":
            parts.append(" ")
            continue
        if kind == "op":
            parts.append(text if _is_unary(tokens, i) else f" {text} ")
        elif kind == "fraction" and previous == "number":
            parts.append(" " + text)  # mixed number: 2 3/4
        elif kind == "punct" and text in (",", ";", ":"):
            parts.append(text + " ")
        else:
            parts.append(text)
        previous = kind
    out = re.sub(r"\s+", " ", "".join(parts))
    out = re.sub(r"\s+([.,?:;!)\]])", r"\1", out)  # no space before punctuation / closing
    out = re.sub(r"([(\[])\s+", r"\1", out)  # no space after opening bracket
    return out.strip()


def _row(e: ET.Element) -> str:
    return _render_row(_tokens(e))


# --- parentheses for fraction parts and powers -------------------------------------------


def _unwrap(e: ET.Element) -> ET.Element:
    """Descend through rows that hold a single child: <mrow><mn>27</mn></mrow> is just 27."""
    while _name(e) in ("mrow", "mstyle") and len(e) == 1:
        e = e[0]
    return e


def _is_enclosed(text: str) -> bool:
    """True if the outer brackets of `text` match each other, as in "(1 + 2)"."""
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(text) - 1:
            return False
    return True


def _needs_parentheses(e: ET.Element) -> bool:
    e = _unwrap(e)
    name = _name(e)
    if name == "mfrac":
        return True
    if name not in ("mrow", "mstyle"):
        return False
    tokens = _tokens(e)
    significant = [i for i, (k, _) in enumerate(tokens) if k != "space"]
    binary = any(tokens[i][0] == "op" and not _is_unary(tokens, i) for i in significant)
    operands = sum(1 for i in significant if tokens[i][0] in ("operand", "number", "fraction"))
    return binary or operands > 1


def _part(e: ET.Element) -> str:
    text = math_to_text(e)
    if _needs_parentheses(e) and not _is_enclosed(text):
        return f"({text})"
    return text


def _two_children(e: ET.Element) -> tuple[ET.Element, ET.Element]:
    if len(e) != 2:
        raise UnsupportedMathML(f"<{_name(e)}> must have 2 children, found {len(e)}")
    return e[0], e[1]


# --- element handlers --------------------------------------------------------------------


def _mfrac(e: ET.Element) -> str:
    numerator, denominator = _two_children(e)
    return f"{_part(numerator)}/{_part(denominator)}"


def _msup(e: ET.Element) -> str:
    base, exponent = _two_children(e)
    return f"{_part(base)}^{_part(exponent)}"


def _munder(e: ET.Element) -> str:
    base, under = _two_children(e)
    if _name(under) == "mo" and set(_text(under)) == {"_"}:
        return "___"  # fill-in blank: <munder><mspace/><mo>__</mo></munder>
    return f"{math_to_text(base)} {math_to_text(under)}".strip()


def _mover(e: ET.Element) -> str:
    base, over = _two_children(e)
    if _name(over) == "mo" and _text(over) in ("–", "-", "¯", "‾", "—"):
        # Overline on digits marks a repeating decimal: 0.3̅
        return "".join(ch + _OVERLINE for ch in math_to_text(base))
    return f"{math_to_text(base)} {math_to_text(over)}".strip()


def _menclose(e: ET.Element) -> str:
    inner = _row(e)
    if e.get("notation") == "longdiv":
        return "⟌" + inner
    return inner  # strikes used to show cancelled common factors: keep the content


def _mtable(e: ET.Element) -> str:
    rows: list[str] = []
    for tr in e:
        if _name(tr) != "mtr":
            raise UnsupportedMathML(f"<mtable> child <{_name(tr)}>")
        cells = [_row(td) for td in tr if _name(td) == "mtd"]
        rows.append(" ".join(c for c in cells if c))
    return "; ".join(r for r in rows if r)


_HANDLERS: dict[str, Callable[[ET.Element], str]] = {
    "math": _row,
    "mrow": _row,
    "mstyle": _row,
    "mtd": _row,
    "mn": _text,
    "mi": _text,
    "mtext": _text,
    "mo": _operator,
    "mspace": lambda e: " ",
    "mfrac": _mfrac,
    "msup": _msup,
    "munder": _munder,
    "mover": _mover,
    "menclose": _menclose,
    "mtable": _mtable,
}


def math_to_text(e: ET.Element) -> str:
    """Render a MathML element (usually <m:math>) as one line of text."""
    handler = _HANDLERS.get(_name(e))
    if handler is None:
        raise UnsupportedMathML(f"unsupported MathML element <{_name(e)}>")
    return handler(e)


# --- scope checks (G1: positive fractions only, no variables) ----------------------------

_ROW_LIKE = ("math", "mrow", "mstyle", "mtd", "menclose")


def has_negative(e: ET.Element) -> bool:
    """True if any number is negative: a unary minus, or a minus written inside an <mn>."""
    for node in e.iter():
        name = _name(node)
        if name == "mn" and _text(node).startswith(("−", "-", "–")):
            return True
        if name in _ROW_LIKE:
            tokens = _tokens(node)
            if any(_is_unary(tokens, i) and tokens[i][1] == MINUS for i in range(len(tokens))):
                return True
    return False


def has_variable(e: ET.Element) -> bool:
    """True if an identifier is a letter other than a known constant (π)."""
    return any(
        _name(node) == "mi" and _text(node).isalpha() and _text(node) not in _CONSTANTS
        for node in e.iter()
    )
