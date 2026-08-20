from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestTextRequest(BaseModel):
    source: str = Field(..., description="Human-readable identifier for the document, e.g. a filename or URL.")
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    source: str
    chunk_ids: list[str]
    chunk_count: int


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    dense_k: int = Field(20, ge=1, le=200)
    bm25_k: int = Field(20, ge=1, le=200)
    rerank: bool = True


class ChunkResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    source: str
    chunk_index: int
    text: str
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    stages: dict[str, Any]
    results: list[ChunkResult]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    provider: Literal["local", "openai", "anthropic"] | None = Field(
        default=None, description="Overrides DEFAULT_LLM_PROVIDER for this request."
    )


class SourceRef(BaseModel):
    id: int = Field(..., description="1-indexed position - matches the [n] citation markers in the answer.")
    chunk_id: str
    document_id: str
    source: str
    chunk_index: int
    text: str


class CitationReport(BaseModel):
    valid: bool
    cited_sources: list[int]
    coverage: float = Field(..., description="Fraction of retrieved sources actually cited in the answer.")
    issues: list[str]
    per_citation: list[dict[str, Any]]


class GenerateResponse(BaseModel):
    query: str
    answer: str
    provider_used: str
    sources: list[SourceRef]
    citation_report: CitationReport


# ---------------------------------------------------------------------------
# Citation validation (standalone)
# ---------------------------------------------------------------------------
class CitationValidateRequest(BaseModel):
    answer: str
    sources: list[SourceRef]
    support_threshold: float | None = None
