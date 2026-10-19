from learning_agent.kb.chunker import Chunk
from learning_agent.kb.embedder import HashEmbedder
from learning_agent.kb.ingest import document_text, embed_chunks


def chunk(id_: str, text: str, *, in_scope: bool = True) -> Chunk:
    return Chunk(
        id=id_,
        module_id="m1",
        section_title="Divide Fractions",
        doc_type="theory",
        text=text,
        in_scope=in_scope,
        scope_reason=None if in_scope else "negative_numbers",
    )


CHUNKS = [
    chunk("m1:p1", "one two three"),
    chunk("m1:p2", "one two three four five six"),
    chunk("m1:p3", "a negative fraction", in_scope=False),
]


def test_document_text_prepends_the_section_title_when_configured() -> None:
    assert document_text(CHUNKS[0], True) == "Divide Fractions\none two three"
    assert document_text(CHUNKS[0], False) == "one two three"


def test_only_in_scope_chunks_are_embedded_with_the_text_that_is_configured() -> None:
    e = HashEmbedder(dim=4)
    result = embed_chunks(e, CHUNKS, include_section_title=True, max_tokens=100)
    assert list(result.vectors) == ["m1:p1", "m1:p2"]
    expected = e.embed_query("Divide Fractions\none two three")
    assert (result.vectors["m1:p1"] == expected).all()


def test_chunks_longer_than_the_model_limit_are_reported() -> None:
    # HashEmbedder counts words; with the title, m1:p1 has 5 and m1:p2 has 8
    result = embed_chunks(HashEmbedder(dim=4), CHUNKS, include_section_title=True, max_tokens=5)
    assert result.longest_tokens == 8
    assert result.truncated == ["m1:p2"]
