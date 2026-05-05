from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from fastapi import UploadFile

from app.config import Settings
from app.rag.chunker import split_text
from app.rag.llm import LLMService
from app.rag.vector_store import VectorStore


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class DocumentIngestor:
    def __init__(self, settings: Settings, llm: LLMService, store: VectorStore):
        self.settings = settings
        self.llm = llm
        self.store = store

    async def ingest_upload(self, upload: UploadFile) -> dict:
        raw = await upload.read()
        if len(raw) > self.settings.max_upload_bytes:
            raise ValueError(f"File is larger than {self.settings.max_upload_mb} MB.")

        filename = Path(upload.filename or "untitled.txt").name
        text = extract_text(filename, raw)
        chunks = split_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No readable text was found in the uploaded file.")

        embeddings = self.llm.embed_texts(chunks)
        return self.store.add_document(
            name=filename,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self.settings.openai_embedding_model,
            metadata={
                "filename": filename,
                "content_type": upload.content_type or "",
                "size_bytes": len(raw),
            },
        )


def extract_text(filename: str, raw: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}. Use {supported}.")

    if extension in {".txt", ".md", ".markdown"}:
        return raw.decode("utf-8", errors="ignore")
    if extension == ".pdf":
        return _extract_pdf(raw)
    if extension == ".docx":
        return _extract_docx(raw)
    raise ValueError(f"Unsupported file type: {extension}")


def _extract_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(raw: bytes) -> str:
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        if "".join(texts).strip():
            paragraphs.append("".join(texts))
    return "\n\n".join(paragraphs)

