from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Source(BaseModel):
    id: str
    document_id: str
    document_name: str
    chunk_index: int
    score: float
    text: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list)
    use_rag: bool = True
    top_k: int | None = Field(default=None, ge=0, le=12)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    used_rag: bool


class DocumentInfo(BaseModel):
    id: str
    name: str
    created_at: str
    chunk_count: int
    metadata: dict


class UploadResponse(BaseModel):
    document: DocumentInfo
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: list[Source]


class HealthResponse(BaseModel):
    ok: bool
    app: str
    chat_model: str
    embedding_model: str
    api_key_configured: bool
    documents: int
    chunks: int

