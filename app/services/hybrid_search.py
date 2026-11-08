"""
Three-stage hybrid search:

  1. Dense retrieval   - embed the query, ANN search the vector store
  2. BM25 retrieval    - lexical search over the same corpus
  3. Fusion + rerank   - Reciprocal Rank Fusion merges the two candidate
                          sets, then a reranker re-scores the fused
                          candidates for the final top-k

Dense and BM25 catch different failure modes (semantic paraphrase vs. exact
keyword/acronym/ID matches); fusing them before reranking gives the
reranker a broader, higher-recall candidate pool than either stage alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.bm25_index import CorpusStore
from app.services.embeddings import EmbeddingProvider
from app.services.fusion import reciprocal_rank_fusion
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore


@dataclass
class HybridSearchResult:
    chunk_id: str
    document_id: str
    source: str
    chunk_index: int
    text: str
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float | None
    rerank_score: float | None


class HybridSearchService:
    def __init__(
        self,
        corpus: CorpusStore,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        rrf_k: int = 60,
    ):
        self._corpus = corpus
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        bm25_k: int = 20,
        rerank: bool = True,
    ) -> tuple[list[HybridSearchResult], dict[str, Any]]:
        # --- stage 1: dense ---
        query_vector = self._embedder.embed_one(query)
        dense_hits = self._vector_store.search(query_vector, top_k=dense_k)
        dense_ranked = [(h.chunk_id, h.score) for h in dense_hits]
        dense_scores = {cid: score for cid, score in dense_ranked}

        # --- stage 2: bm25 ---
        bm25_ranked = self._corpus.search(query, top_k=bm25_k)
        bm25_scores = {cid: score for cid, score in bm25_ranked}

        # --- fuse ---
        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=self._rrf_k)
        fused_ranked_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])]

        candidate_pool_size = max(top_k, dense_k, bm25_k) if rerank else top_k
        candidate_ids = fused_ranked_ids[:candidate_pool_size]
        candidate_chunks = self._corpus.get_many(candidate_ids)
        chunks_by_id = {c.chunk_id: c for c in candidate_chunks}

        # --- stage 3: rerank ---
        rerank_scores: dict[str, float] = {}
        if rerank and candidate_chunks:
            reranked = self._reranker.rerank(
                query, [(c.chunk_id, c.text) for c in candidate_chunks], top_k=top_k
            )
            final_order = [r.chunk_id for r in reranked]
            rerank_scores = {r.chunk_id: r.score for r in reranked}
        else:
            final_order = candidate_ids[:top_k]

        results: list[HybridSearchResult] = []
        for cid in final_order:
            chunk = chunks_by_id.get(cid) or self._corpus.get(cid)
            if chunk is None:
                continue
            results.append(
                HybridSearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    dense_score=dense_scores.get(cid),
                    bm25_score=bm25_scores.get(cid),
                    rrf_score=fused.get(cid),
                    rerank_score=rerank_scores.get(cid),
                )
            )

        stage_stats = {
            "dense_candidates": len(dense_ranked),
            "bm25_candidates": len(bm25_ranked),
            "fused_candidates": len(fused_ranked_ids),
            "reranked": rerank,
        }
        return results, stage_stats
