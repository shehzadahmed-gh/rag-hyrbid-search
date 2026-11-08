from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.routers import citations, generation, ingestion, retrieval

settings = get_settings()

app = FastAPI(
    title="RAG Pipeline with Hybrid Search",
    description=(
        "Three-stage hybrid search (dense vector retrieval + BM25 + reranking) over "
        "internal documents, with grounded answer generation and citation validation."
    ),
    version="1.0.0",
)

app.include_router(ingestion.router)
app.include_router(retrieval.router)
app.include_router(generation.router)
app.include_router(citations.router)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": settings.app_name,
        "status": "ok",
        "vector_backend": settings.vector_backend,
        "embedding_backend": settings.embedding_backend,
        "reranker_backend": settings.reranker_backend,
        "default_llm_provider": settings.default_llm_provider,
    }


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
