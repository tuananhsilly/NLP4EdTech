import xml.etree.ElementTree as ET

import pytest

from learning_agent.kb.mathml import UnsupportedMathML, has_negative, has_variable, math_to_text


def math(body: str) -> ET.Element:
    """Parse a MathML fragment written with the `m:` prefix, as in CNXML."""
    return ET.fromstring(f'<m:math xmlns:m="http://www.w3.org/1998/Math/MathML">{body}</m:math>')


FRAC_27_32 = "<m:mfrac><m:mrow><m:mn>27</m:mn></m:mrow><m:mrow><m:mn>32</m:mn></m:mrow></m:mfrac>"
FRAC_11_32 = "<m:mfrac><m:mn>11</m:mn><m:mn>32</m:mn></m:mfrac>"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # numbers and simple fractions
        ("<m:mn>27</m:mn>", "27"),
        (FRAC_27_32, "27/32"),
        # binary operator gets spaces; sentence punctuation sticks to the left
        (
            f"<m:mrow>{FRAC_27_32}<m:mo>−</m:mo>{FRAC_11_32}<m:mo>.</m:mo></m:mrow>",
            "27/32 − 11/32.",
        ),
        # compound numerator/denominator are parenthesized
        (
            "<m:mfrac><m:mrow><m:mn>2</m:mn><m:mo>·</m:mo><m:mn>5</m:mn></m:mrow>"
            "<m:mrow><m:mn>3</m:mn><m:mo>·</m:mo><m:mn>5</m:mn></m:mrow></m:mfrac>",
            "(2 × 5)/(3 × 5)",
        ),
        # an explicit pair of parentheses is not doubled
        (
            "<m:mfrac><m:mrow><m:mo>(</m:mo><m:mn>1</m:mn><m:mo>+</m:mo><m:mn>2</m:mn>"
            "<m:mo>)</m:mo></m:mrow><m:mn>3</m:mn></m:mfrac>",
            "(1 + 2)/3",
        ),
        # nested fraction (complex fraction)
        (f"<m:mfrac>{FRAC_11_32}<m:mn>2</m:mn></m:mfrac>", "(11/32)/2"),
        # mixed number: a number directly followed by a fraction
        ("<m:mrow><m:mn>2</m:mn><m:mfrac><m:mn>3</m:mn><m:mn>4</m:mn></m:mfrac></m:mrow>", "2 3/4"),
        # ... but not when an operator is between them
        (
            "<m:mrow><m:mn>2</m:mn><m:mo>⋅</m:mo><m:mfrac><m:mn>3</m:mn><m:mn>4</m:mn></m:mfrac></m:mrow>",
            "2 × 3/4",
        ),
        # unary minus: no space after it, and no parentheses needed in a numerator
        ("<m:mrow><m:mo>−</m:mo><m:mfrac><m:mn>3</m:mn><m:mn>8</m:mn></m:mfrac></m:mrow>", "−3/8"),
        (
            "<m:mfrac><m:mrow><m:mo>−</m:mo><m:mn>3</m:mn></m:mrow><m:mn>8</m:mn></m:mfrac>",
            "−3/8",
        ),
        (
            "<m:mrow><m:mn>5</m:mn><m:mo>+</m:mo><m:mo>(</m:mo><m:mo>-</m:mo><m:mn>3</m:mn>"
            "<m:mo>)</m:mo></m:mrow>",
            "5 + (−3)",
        ),
        # division sign kept (R2-D4); both multiplication dots become ×
        ("<m:mrow><m:mn>6</m:mn><m:mo>÷</m:mo><m:mn>2</m:mn></m:mrow>", "6 ÷ 2"),
        # power
        ("<m:msup><m:mn>2</m:mn><m:mn>3</m:mn></m:msup>", "2^3"),
        # text, spaces and bold styling
        (
            "<m:mrow><m:mtext>half</m:mtext><m:mspace width='1em'/><m:mstyle fontweight='bold'>"
            "<m:mtext>of</m:mtext></m:mstyle><m:mspace width='1em'/><m:mn>8</m:mn></m:mrow>",
            "half of 8",
        ),
        # fill-in blank
        ("<m:munder accentunder='true'><m:mspace width='1em'/><m:mo>__</m:mo></m:munder>", "___"),
        # a blank between two numbers stands for a missing relation (<, >, =), so it is
        # spaced like an operator and a minus after it is a sign (found on real data)
        (
            "<m:mrow><m:mn>9</m:mn><m:munder><m:mspace/><m:mo>__</m:mo></m:munder>"
            "<m:mo>−</m:mo><m:mn>3</m:mn></m:mrow>",
            "9 ___ −3",
        ),
        # a colon or comma token is followed by a space (found on real data)
        (
            "<m:mrow><m:mn>3</m:mn><m:mo>:</m:mo><m:mo>(</m:mo><m:mn>1</m:mn><m:mo>)</m:mo></m:mrow>",
            "3: (1)",
        ),
        # long division bracket, cancellation strike
        (
            "<m:mrow><m:mn>3</m:mn>"
            "<m:menclose notation='longdiv'><m:mn>11</m:mn></m:menclose></m:mrow>",
            "3⟌11",
        ),
        ("<m:menclose notation='updiagonalstrike'><m:mn>5</m:mn></m:menclose>", "5"),
        # repeating digit: overline accent
        ("<m:mrow><m:mn>0.</m:mn><m:mover><m:mn>3</m:mn><m:mo>–</m:mo></m:mover></m:mrow>", "0.3̅"),
        # table of text cells
        (
            "<m:mtable><m:mtr><m:mtd><m:mtext>How many parts?</m:mtext></m:mtd><m:mtd/>"
            "<m:mtd><m:mtext>Eight</m:mtext></m:mtd></m:mtr>"
            "<m:mtr><m:mtd><m:mtext>Shaded?</m:mtext></m:mtd><m:mtd><m:mn>3</m:mn></m:mtd></m:mtr>"
            "</m:mtable>",
            "How many parts? Eight; Shaded? 3",
        ),
    ],
)
def test_math_to_text(body: str, expected: str) -> None:
    assert math_to_text(math(body)) == expected


def test_unknown_element_raises_instead_of_guessing() -> None:
    with pytest.raises(UnsupportedMathML, match="msqrt"):
        math_to_text(math("<m:msqrt><m:mn>4</m:mn></m:msqrt>"))


def test_mfrac_with_wrong_arity_raises() -> None:
    with pytest.raises(UnsupportedMathML, match="mfrac"):
        math_to_text(math("<m:mfrac><m:mn>1</m:mn></m:mfrac>"))


# --- scope detection (G1, R2-D3) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "negative"),
    [
        (f"<m:mrow>{FRAC_27_32}<m:mo>−</m:mo>{FRAC_11_32}</m:mrow>", False),  # subtraction
        ("<m:mrow><m:mo>−</m:mo><m:mn>3</m:mn></m:mrow>", True),  # leading sign
        ("<m:mrow><m:mn>5</m:mn><m:mo>−</m:mo><m:mo>−</m:mo><m:mn>3</m:mn></m:mrow>", True),
        ("<m:mrow><m:mn>5</m:mn><m:mo>=</m:mo><m:mo>−</m:mo><m:mn>3</m:mn></m:mrow>", True),
        ("<m:mrow><m:mo>(</m:mo><m:mo>−</m:mo><m:mn>3</m:mn><m:mo>)</m:mo></m:mrow>", True),
        ("<m:mfrac><m:mrow><m:mo>−</m:mo><m:mn>3</m:mn></m:mrow><m:mn>8</m:mn></m:mfrac>", True),
        ("<m:mn>−4</m:mn>", True),  # sign written inside the number
    ],
)
def test_has_negative(body: str, negative: bool) -> None:
    assert has_negative(math(body)) is negative


@pytest.mark.parametrize(
    ("body", "variable"),
    [
        ("<m:mfrac><m:mi>y</m:mi><m:mn>6</m:mn></m:mfrac>", True),
        ("<m:mrow><m:mn>2</m:mn><m:mi>π</m:mi></m:mrow>", False),  # π is a constant
        ("<m:mtext>half of</m:mtext>", False),  # words are not variables
        (FRAC_11_32, False),
    ],
)
def test_has_variable(body: str, variable: bool) -> None:
    assert has_variable(math(body)) is variable
