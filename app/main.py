from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.rag.ingest import DocumentIngestor
from app.rag.llm import LLMService, MissingOpenAIKeyError
from app.rag.vector_store import SearchHit, VectorStore
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    Source,
    UploadResponse,
)


settings = get_settings()
store = VectorStore(settings.vector_store_path)
llm = LLMService(settings)
ingestor = DocumentIngestor(settings=settings, llm=llm, store=store)

app = FastAPI(
    title="Simple RAG ChatGPT",
    description="A compact ChatGPT-style application with OpenAI API + local RAG.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    stats = store.stats()
    return HealthResponse(
        ok=True,
        app="Simple RAG ChatGPT",
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
        api_key_configured=llm.is_configured,
        documents=stats["documents"],
        chunks=stats["chunks"],
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**document) for document in store.list_documents()]


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    try:
        document = await ingestor.ingest_upload(file)
    except MissingOpenAIKeyError as error:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is missing. Configure .env before uploading documents.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Document ingestion failed: {error}") from error

    return UploadResponse(document=DocumentInfo(**document), chunks=document["chunk_count"])


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    deleted = store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True}


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        query_embedding = llm.embed_texts([request.query])[0]
    except MissingOpenAIKeyError as error:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is missing. Configure .env before searching.",
        ) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {error}") from error

    hits = store.search(query_embedding, top_k=request.top_k)
    return SearchResponse(results=[source_from_hit(hit) for hit in hits])


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    history = [ChatMessage(role=item.role, content=item.content) for item in request.history[-10:]]
    top_k = request.top_k if request.top_k is not None else settings.rag_top_k
    sources: list[Source] = []

    try:
        if request.use_rag and store.stats()["chunks"] > 0 and top_k > 0:
            query_embedding = llm.embed_texts([request.message])[0]
            sources = [source_from_hit(hit) for hit in store.search(query_embedding, top_k=top_k)]

        answer = llm.answer(
            history=history,
            question=request.message,
            sources=sources,
        )
    except MissingOpenAIKeyError as error:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is missing. Configure .env before chatting.",
        ) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {error}") from error

    return ChatResponse(
        answer=answer,
        sources=sources,
        model=settings.openai_chat_model,
        used_rag=bool(sources),
    )


def source_from_hit(hit: SearchHit) -> Source:
    return Source(
        id=hit.id,
        document_id=hit.document_id,
        document_name=hit.document_name,
        chunk_index=hit.chunk_index,
        score=round(hit.score, 4),
        text=hit.text,
    )

