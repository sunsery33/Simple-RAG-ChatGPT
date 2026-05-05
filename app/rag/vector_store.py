from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class SearchHit:
    id: str
    document_id: str
    document_name: str
    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def add_document(
        self,
        name: str,
        chunks: list[str],
        embeddings: list[list[float]],
        embedding_model: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            raise ValueError("document contains no chunks")

        with self._lock:
            current_model = self._data.get("embedding_model")
            if self._data["chunks"] and current_model != embedding_model:
                raise ValueError(
                    "Existing vector store was built with "
                    f"{current_model}; reset it before using {embedding_model}."
                )

            document_id = str(uuid4())
            now = datetime.now(UTC).isoformat()
            document = {
                "id": document_id,
                "name": name,
                "created_at": now,
                "chunk_count": len(chunks),
                "metadata": metadata or {},
            }

            chunk_records = []
            for index, (text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                chunk_records.append(
                    {
                        "id": str(uuid4()),
                        "document_id": document_id,
                        "document_name": name,
                        "chunk_index": index,
                        "text": text,
                        "embedding": embedding,
                        "metadata": metadata or {},
                    }
                )

            self._data["embedding_model"] = embedding_model
            self._data["documents"].append(document)
            self._data["chunks"].extend(chunk_records)
            self._save_unlocked()
            return document

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(document) for document in self._data["documents"]]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            before = len(self._data["documents"])
            self._data["documents"] = [
                document
                for document in self._data["documents"]
                if document["id"] != document_id
            ]
            self._data["chunks"] = [
                chunk for chunk in self._data["chunks"] if chunk["document_id"] != document_id
            ]
            deleted = len(self._data["documents"]) != before
            if deleted:
                if not self._data["chunks"]:
                    self._data["embedding_model"] = None
                self._save_unlocked()
            return deleted

    def reset(self) -> None:
        with self._lock:
            self._data = self._empty()
            self._save_unlocked()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "documents": len(self._data["documents"]),
                "chunks": len(self._data["chunks"]),
            }

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchHit]:
        if top_k <= 0:
            return []

        with self._lock:
            candidates = list(self._data["chunks"])

        hits = []
        for chunk in candidates:
            score = cosine_similarity(query_embedding, chunk["embedding"])
            hits.append(
                SearchHit(
                    id=chunk["id"],
                    document_id=chunk["document_id"],
                    document_name=chunk["document_name"],
                    chunk_index=chunk["chunk_index"],
                    score=score,
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        data.setdefault("version", 1)
        data.setdefault("embedding_model", None)
        data.setdefault("documents", [])
        data.setdefault("chunks", [])
        return data

    def _save_unlocked(self) -> None:
        temp_path = self.path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=2)
        temp_path.replace(self.path)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": 1,
            "embedding_model": None,
            "documents": [],
            "chunks": [],
        }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

