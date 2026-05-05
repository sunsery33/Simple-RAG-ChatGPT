import re


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_oversized_unit(paragraph, chunk_size, overlap))

    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = unit

    if current:
        chunks.append(current)

    return chunks


def _split_oversized_unit(unit: str, chunk_size: int, overlap: int) -> list[str]:
    if len(unit) <= chunk_size:
        return [unit]

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？.!?])\s+", unit)
        if part.strip()
    ]

    if len(sentence_parts) > 1 and all(len(part) <= chunk_size for part in sentence_parts):
        return _pack_sentences(sentence_parts, chunk_size)

    step = chunk_size - overlap
    return [unit[start : start + chunk_size].strip() for start in range(0, len(unit), step)]


def _pack_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence

    if current:
        chunks.append(current)

    return chunks
