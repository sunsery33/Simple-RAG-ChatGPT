from app.rag.chunker import clean_text, split_text


def test_clean_text_normalizes_spacing():
    text = "第一行  \r\n\r\n\r\n  第二行"
    assert clean_text(text) == "第一行\n\n第二行"


def test_split_text_returns_paragraph_chunks():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = split_text(text, chunk_size=20, overlap=4)
    assert chunks
    assert "第一段内容" in chunks[0]


def test_split_text_overlaps_long_text():
    text = "a" * 320
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0][-20:] == chunks[1][:20]

