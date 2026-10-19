import xml.etree.ElementTree as ET

import pytest

from learning_agent.kb.chunker import Chunk, UnsupportedCNXML, chunk_module
from learning_agent.kb.manifest import ModuleEntry

NS = 'xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML"'
F = "<m:math><m:mfrac><m:mn>{}</m:mn><m:mn>{}</m:mn></m:mfrac></m:math>"  # fraction a/b


def frac(a: int, b: int) -> str:
    return F.format(a, b)


# A hand-written miniature of a Prealgebra module; each block exercises one rule.
DOC = f"""<document {NS}>
<title>Add Fractions</title>
<content>
 <note id="prep" class="be-prepared"><para id="p0">Before you get started...</para></note>
 <section id="s1">
  <title>Add Fractions with a Common Denominator</title>
  <para id="p1">To add {frac(1, 5)} and {frac(2, 5)}, <emphasis>add the numerators</emphasis>
   and keep the denominator.</para>
  <note id="def1"><title>Fraction Addition</title>Add the numerators, keep the denominator.</note>
  <figure id="fig1"><media id="img1" alt="A bar split into five equal parts.">
   <image src="x.png"/></media>
   <caption>Fifths</caption></figure>
  <note id="try1" class="try"><exercise id="t1"><problem id="tp1"><para id="tp1p">Add.</para>
   </problem></exercise></note>
  <example id="ex1"><exercise id="ex1e">
   <problem id="ex1p"><para id="ex1pp">Find the sum: {frac(1, 5)} + {frac(2, 5)}.</para></problem>
   <solution id="ex1s"><title>Solution</title>
    <table id="tab1" aria-label="Shows one fifth plus two fifths being combined into three fifths.">
     <tgroup cols="2"><tbody>
      <row><entry>Add the numerators.</entry><entry>
       <media id="img2" alt="WRONG ALT: x over y"><image src="y.png"/></media></entry></row>
      <row><entry>Simplify.</entry><entry>{frac(3, 5)}</entry></row>
     </tbody></tgroup></table>
   </solution></exercise></example>
  <note id="how1" class="howto"><title>Add fractions with a common denominator.</title>
   <list id="l1" list-type="enumerated"><item>Add the numerators.</item>
    <item>Keep the denominator.</item></list></note>
  <para id="p2">Next paragraph after the example.</para>
  <table id="tab2" summary="Fraction names"><tgroup cols="2"><tbody>
   <row><entry>{frac(1, 2)}</entry><entry>one half</entry></row></tbody></tgroup></table>
  <list id="l2"><item>Halves</item><item>Thirds</item></list>
 </section>
 <section id="s2">
  <title>Subtract Negative Fractions</title>
  <para id="p3">A negative fraction is less than zero.</para>
  <example id="ex2"><exercise id="ex2e"><problem id="ex2p"><para id="ex2pp">Find
   <m:math><m:mrow><m:mo>−</m:mo><m:mfrac><m:mn>3</m:mn><m:mn>8</m:mn></m:mfrac></m:mrow></m:math>.
   </para></problem>
   <solution id="ex2s"><para id="ex2sp">Done.</para></solution></exercise></example>
  <example id="ex3"><exercise id="ex3e"><problem id="ex3p"><para id="ex3pp">Simplify
   <m:math><m:mfrac><m:mi>y</m:mi><m:mn>6</m:mn></m:mfrac></m:math>.</para></problem>
   <solution id="ex3s"><para id="ex3sp">It is already simplified.</para></solution>
  </exercise></example>
 </section>
 <section id="s4"><title>Fraction Addition</title>
  <para id="p5">If <m:math><m:mi>c</m:mi><m:mo>≠</m:mo><m:mn>0</m:mn></m:math>, then
   <m:math><m:mrow><m:mfrac><m:mi>a</m:mi><m:mi>c</m:mi></m:mfrac><m:mo>+</m:mo>
   <m:mfrac><m:mi>b</m:mi><m:mi>c</m:mi></m:mfrac></m:mrow></m:math> adds the numerators.</para>
 </section>
 <section id="s3"><title>Simplify Complex Fractions</title>
  <para id="p4">A complex fraction has a fraction in the numerator or denominator.</para>
 </section>
 <section id="kc" class="key-concepts"><title>Key Concepts</title>
  <list id="kcl"><item>To add fractions with a common denominator, add the numerators.</item></list>
 </section>
 <section id="ex" class="section-exercises"><section id="pp" class="practice-perfect">
  <title>Practice Makes Perfect</title><exercise id="e1"><problem id="e1p"><para id="e1pp">1 + 1
  </para></problem></exercise></section></section>
</content>
<glossary><definition id="d1"><term>mixed number</term>
 <meaning id="d1m">A mixed number consists of a whole number and a proper fraction.</meaning>
</definition>
<definition id="d2"><term>reciprocal</term><meaning id="d2m">The reciprocal of
 <m:math><m:mfrac><m:mi>a</m:mi><m:mi>b</m:mi></m:mfrac></m:math> is
 <m:math><m:mfrac><m:mi>b</m:mi><m:mi>a</m:mi></m:mfrac></m:math>.</meaning></definition></glossary>
</document>"""

MODULE = ModuleEntry(
    id="m1",
    section="4.4",
    title="Add Fractions",
    scope="partial",
    exclude_sections=("Simplify Complex Fractions",),
    sha256="0" * 64,
)


def chunks(doc: str = DOC, module: ModuleEntry = MODULE, max_words: int = 250) -> list[Chunk]:
    return chunk_module(ET.fromstring(doc), module, max_words=max_words)


def by_id(items: list[Chunk]) -> dict[str, Chunk]:
    return {c.id: c for c in items}


# --- which blocks become chunks (R2-D1) --------------------------------------------------


def test_block_types_and_ids() -> None:
    result = by_id(chunks())
    assert {cid: c.doc_type for cid, c in result.items()} == {
        "m1:p1": "theory",  # p1 + def1 + fig1 grouped until the example
        "m1:ex1": "worked_example",
        "m1:how1": "procedure",
        "m1:p2": "theory",
        "m1:p3": "theory",
        "m1:ex2": "worked_example",
        "m1:ex3": "worked_example",
        "m1:p4": "theory",
        "m1:kc": "key_concepts",
        "m1:p5": "theory",
        "m1:d1": "definition",
        "m1:d2": "definition",
    }


def test_try_it_be_prepared_and_exercise_sections_are_skipped() -> None:
    text = " ".join(c.text for c in chunks())
    assert "Before you get started" not in text
    assert "Add." not in text.split()
    assert "1 + 1" not in text


def test_section_title_is_recorded() -> None:
    assert by_id(chunks())["m1:ex1"].section_title == "Add Fractions with a Common Denominator"


# --- text rendering ----------------------------------------------------------------------


def test_text_after_inline_math_and_title_is_kept() -> None:
    theory = by_id(chunks())["m1:p1"].text
    # tail text after <m:math> and after <emphasis>, and the text after <title> in a note
    assert "To add 1/5 and 2/5, add the numerators and keep the denominator." in theory
    assert "Fraction Addition" in theory
    assert "Add the numerators, keep the denominator." in theory


def test_worked_example_has_problem_and_solution_steps() -> None:
    text = by_id(chunks())["m1:ex1"].text
    assert text.startswith("Example: Find the sum: 1/5 + 2/5.")
    assert "Solution:" in text
    assert "Simplify: 3/5" in text


def test_image_alt_in_solution_table_is_not_used_but_table_description_is() -> None:
    # R2-D2: alt texts inside step tables were found wrong; use the table's aria-label.
    text = by_id(chunks())["m1:ex1"].text
    assert "WRONG ALT" not in text
    assert "Shows one fifth plus two fifths being combined into three fifths." in text


def test_figure_alt_and_caption_are_kept_outside_tables() -> None:
    text = by_id(chunks())["m1:p1"].text
    assert "[Figure: A bar split into five equal parts.]" in text
    assert "Fifths" in text


def test_lists_become_lines() -> None:
    text = by_id(chunks())["m1:how1"].text
    assert "- Add the numerators.\n- Keep the denominator." in text


def test_table_and_list_directly_in_a_section_are_rendered_as_blocks() -> None:
    # Found on real data: a section-level <table> was rendered from its <tgroup> child.
    assert by_id(chunks())["m1:p2"].text == (
        "Next paragraph after the example.\n1/2: one half\n- Halves\n- Thirds"
    )


def test_definition_chunk_reads_term_and_meaning() -> None:
    assert by_id(chunks())["m1:d1"].text == (
        "mixed number: A mixed number consists of a whole number and a proper fraction."
    )


def test_theory_is_split_at_paragraph_boundaries_when_too_long() -> None:
    long_doc = (
        f"<document {NS}><title>T</title><content><section id='s'><title>S</title>"
        + "".join(f"<para id='q{i}'>{'word ' * 30}</para>" for i in range(5))
        + "</section></content></document>"
    )
    result = chunks(long_doc, max_words=70)
    assert [c.id for c in result] == ["m1:q0", "m1:q2", "m1:q4"]
    assert all(len(c.text.split()) <= 70 for c in result)


# --- scope flags (G1, R2-D3) -------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_id", "in_scope", "reason"),
    [
        ("m1:ex1", True, None),
        ("m1:ex2", False, "negative_numbers"),  # unary minus in MathML
        ("m1:p3", False, "negative_numbers"),  # the word "negative" in text
        ("m1:ex3", False, "variables"),  # y/6
        ("m1:p4", False, "excluded_section"),  # listed in exclude_sections
        ("m1:d1", True, None),
        # RV1: letters as general notation are allowed outside worked examples
        ("m1:p5", True, None),
        ("m1:d2", True, None),
    ],
)
def test_scope_flags(chunk_id: str, in_scope: bool, reason: str | None) -> None:
    c = by_id(chunks())[chunk_id]
    assert (c.in_scope, c.scope_reason) == (in_scope, reason)


def test_chunks_listed_in_exclude_chunks_are_excluded() -> None:
    # R3-D1: a glossary term or Key Concepts section can belong to an excluded section
    module = MODULE.model_copy(update={"exclude_chunks": ("d1", "kc")})
    result = by_id(chunks(module=module))
    assert result["m1:d1"].scope_reason == "excluded_section"
    assert result["m1:kc"].scope_reason == "excluded_section"
    assert result["m1:d2"].in_scope


def test_exclude_chunk_that_matches_nothing_raises() -> None:
    module = MODULE.model_copy(update={"exclude_chunks": ("d1", "gone")})
    with pytest.raises(ValueError, match="gone"):
        chunks(module=module)


def test_module_out_flags_every_chunk() -> None:
    out = MODULE.model_copy(update={"scope": "out"})
    result = chunks(module=out)
    assert result and all(c.scope_reason == "module_out" for c in result)


# --- robustness --------------------------------------------------------------------------


def test_unknown_cnxml_element_raises() -> None:
    doc = (
        f"<document {NS}><title>T</title><content><section id='s'><title>S</title>"
        "<para id='p'>Hi <blink>there</blink></para></section></content></document>"
    )
    with pytest.raises(UnsupportedCNXML, match="blink"):
        chunks(doc)


def test_output_is_deterministic_and_ids_are_unique() -> None:
    first, second = chunks(), chunks()
    assert first == second
    assert len({c.id for c in first}) == len(first)
