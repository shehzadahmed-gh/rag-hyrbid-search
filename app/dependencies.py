"""Builds each service once per process from Settings and hands them to the
routers as FastAPI dependencies. Routers only see the interfaces, so swapping
a backend is a .env change.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.bm25_index import CorpusStore
from app.services.embeddings import build_embedding_provider
from app.services.generation_service import GenerationService
from app.services.hybrid_search import HybridSearchService
from app.services.ingestion_service import IngestionService
from app.services.llm_provider import build_llm_provider
from app.services.reranker import build_reranker
from app.services.support_checkers import SupportChecker, build_support_checker
from app.services.vector_store import VectorStore, build_vector_store


@lru_cache
def get_corpus_store() -> CorpusStore:
    settings = get_settings()
    return CorpusStore(corpus_path=settings.corpus_path)


@lru_cache
def get_vector_store() -> VectorStore:
    return build_vector_store(get_settings())


@lru_cache
def get_hybrid_search_service() -> HybridSearchService:
    settings = get_settings()
    return HybridSearchService(
        corpus=get_corpus_store(),
        vector_store=get_vector_store(),
        embedder=build_embedding_provider(settings),
        reranker=build_reranker(settings),
        rrf_k=settings.rrf_k,
        dense_k=settings.dense_top_k,
        bm25_k=settings.bm25_top_k,
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        corpus=get_corpus_store(),
        vector_store=get_vector_store(),
        embedder=build_embedding_provider(settings),
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
    )


@lru_cache
def get_support_checker() -> SupportChecker:
    settings = get_settings()
    return build_support_checker(settings, llm_factory=lambda: build_llm_provider(settings))


@lru_cache
def get_generation_service() -> GenerationService:
    settings = get_settings()
    return GenerationService(
        search_service=get_hybrid_search_service(),
        support_checker=get_support_checker(),
        support_threshold=settings.citation_support_threshold,
    )


def get_llm_provider(provider_override: str | None, settings: Settings | None = None):
    settings = settings or get_settings()
    return build_llm_provider(settings, override=provider_override)
