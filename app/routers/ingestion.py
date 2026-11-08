from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_corpus_store, get_ingestion_service
from app.models.schemas import DocumentSummary, IngestResponse, IngestTextRequest
from app.services.bm25_index import CorpusStore
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/documents", response_model=IngestResponse, status_code=201)
def ingest_document(
    request: IngestTextRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    try:
        document_id, chunk_ids = service.ingest_text(request.source, request.text, request.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(
        document_id=document_id, source=request.source, chunk_ids=chunk_ids, chunk_count=len(chunk_ids)
    )


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(corpus: CorpusStore = Depends(get_corpus_store)) -> list[DocumentSummary]:
    return [DocumentSummary(**doc) for doc in corpus.documents().values()]


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, service: IngestionService = Depends(get_ingestion_service)) -> dict:
    deleted = service.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for document_id={document_id!r}")
    return {"document_id": document_id, "deleted_chunks": deleted}
