"""
Central configuration for the RAG pipeline service.

Every external dependency (vector store backend, embedding provider, LLM
provider, reranker) is chosen here via environment variables so the exact
same codebase can run:

  * fully offline, with zero API keys and zero network access, for local
    development, unit tests, and CI (the defaults below), or
  * against real backends (a Qdrant/Chroma server via docker-compose, and
    GPT-4o / Claude Sonnet) once the corresponding keys / URLs are set.

See .env.example for the full list of variables.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- service ---
    app_name: str = "rag-hybrid-search"
    environment: str = "development"

    # --- chunking ---
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40

    # --- embeddings ---
    # "local"  -> offline hashing-vectorizer embeddings (no network, no key, deterministic)
    # "openai" -> OpenAI text-embedding-3-* (needs OPENAI_API_KEY)
    embedding_backend: Literal["local", "openai"] = "local"
    embedding_dimension: int = 384
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None

    # --- vector store ---
    # "qdrant" or "chroma". Each supports either a real server (URL) or an
    # embedded on-disk mode (LOCAL_PATH) that needs no server at all - handy
    # for local dev/tests and for this take-home style build environment.
    vector_backend: Literal["qdrant", "chroma"] = "qdrant"
    collection_name: str = "documents"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: str | None = "./data/qdrant_local"

    chroma_url: str | None = None
    chroma_local_path: str | None = "./data/chroma_local"

    # --- BM25 / corpus persistence ---
    corpus_path: str = "./data/corpus.jsonl"

    # --- hybrid search ---
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

    # --- reranking ---
    # "local" -> offline TF-IDF/cosine lexical reranker (no network, no key)
    # "cross_encoder" -> sentence-transformers cross-encoder (downloads weights
    #                     from Hugging Face on first use - needs network)
    reranker_backend: Literal["local", "cross_encoder"] = "local"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- generation ---
    # "local"     -> deterministic extractive stub, no API key needed
    # "openai"    -> GPT-4o (needs OPENAI_API_KEY)
    # "anthropic" -> Claude Sonnet (needs ANTHROPIC_API_KEY)
    default_llm_provider: Literal["local", "openai", "anthropic"] = "local"
    openai_chat_model: str = "gpt-4o"
    anthropic_api_key: str | None = None
    # Update as newer Claude Sonnet snapshots ship - see docs.claude.com/en/docs/about-claude/models
    anthropic_chat_model: str = "claude-sonnet-5"
    generation_max_tokens: int = 800

    # --- citation validation ---
    citation_support_threshold: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()
