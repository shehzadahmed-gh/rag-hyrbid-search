from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_hybrid_search_service
from app.models.schemas import ChunkResult, SearchRequest, SearchResponse
from app.services.hybrid_search import HybridSearchService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: HybridSearchService = Depends(get_hybrid_search_service),
) -> SearchResponse:
    results, stage_stats = service.search(
        query=request.query,
        top_k=request.top_k,
        dense_k=request.dense_k,
        bm25_k=request.bm25_k,
        rerank=request.rerank,
    )
    chunk_results = [
        ChunkResult(
            rank=i + 1,
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            source=r.source,
            chunk_index=r.chunk_index,
            text=r.text,
            dense_score=r.dense_score,
            bm25_score=r.bm25_score,
            rrf_score=r.rrf_score,
            rerank_score=r.rerank_score,
        )
        for i, r in enumerate(results)
    ]
    return SearchResponse(query=request.query, stages=stage_stats, results=chunk_results)
