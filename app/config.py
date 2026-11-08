"""Service configuration, read from environment variables or a .env file.

The defaults run fully offline: hashing embeddings, an on-disk Qdrant
collection, the TF-IDF reranker, the extractive answer generator and the
lexical citation checker. See .env.example for every option.
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
    # "qdrant" or "chroma". Set *_URL to use a server, otherwise the
    # embedded on-disk mode at *_LOCAL_PATH is used.
    vector_backend: Literal["qdrant", "chroma"] = "qdrant"
    collection_name: str = "documents"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: str | None = "./data/qdrant_local"

    chroma_url: str | None = None
    chroma_local_path: str | None = "./data/chroma_local"

    # --- BM25 / corpus persistence ---
    corpus_path: str = "./data/corpus.jsonl"

    # --- hybrid search: candidates taken from each retriever before fusion ---
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60

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
    anthropic_chat_model: str = "claude-sonnet-5"
    generation_max_tokens: int = 800

    # --- citation validation ---
    # "lexical" (default, offline), "nli" (cross-encoder entailment model,
    # needs sentence-transformers) or "llm" (uses DEFAULT_LLM_PROVIDER).
    citation_checker: Literal["lexical", "nli", "llm"] = "lexical"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    # Leave unset to use the checker's own default (lexical 0.15, nli/llm 0.5).
    citation_support_threshold: float | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
