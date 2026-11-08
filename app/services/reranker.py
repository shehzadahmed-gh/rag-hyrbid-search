"""
Stage 3 of the hybrid pipeline: rerank the fused dense+BM25 candidate set.

`LocalLexicalReranker` (default) scores each candidate by TF-IDF cosine
similarity to the query - a real, deterministic, offline re-scoring pass
that lets the whole 3-stage pipeline run without a network call.

`CrossEncoderReranker` wraps a `sentence-transformers` cross-encoder for
production-quality semantic reranking. It downloads model weights from
Hugging Face on first use, so it needs outbound network access; import is
deferred so the dependency is optional until this backend is actually
selected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RerankedCandidate:
    chunk_id: str
    score: float


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[RerankedCandidate]:
        """candidates: list of (chunk_id, chunk_text). Returns best-first, length <= top_k."""
        raise NotImplementedError


class LocalLexicalReranker(Reranker):
    name = "local-tfidf"

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[RerankedCandidate]:
        if not candidates:
            return []
        ids = [cid for cid, _ in candidates]
        texts = [text for _, text in candidates]
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            doc_matrix = vectorizer.fit_transform(texts + [query])
        except ValueError:
            # Degenerate corpus (e.g. all-stopword chunks) - fall back to no-op order.
            return [RerankedCandidate(chunk_id=cid, score=0.0) for cid in ids[:top_k]]
        query_vec = doc_matrix[-1]
        doc_vecs = doc_matrix[:-1]
        sims = cosine_similarity(query_vec, doc_vecs)[0]
        ranked = sorted(zip(ids, sims), key=lambda x: -x[1])
        return [RerankedCandidate(chunk_id=cid, score=float(score)) for cid, score in ranked[:top_k]]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # optional dependency

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[RerankedCandidate]:
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip((cid for cid, _ in candidates), scores), key=lambda x: -x[1])
        return [RerankedCandidate(chunk_id=cid, score=float(score)) for cid, score in ranked[:top_k]]


def build_reranker(settings) -> Reranker:
    if settings.reranker_backend == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.cross_encoder_model)
    return LocalLexicalReranker()
