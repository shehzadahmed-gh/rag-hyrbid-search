from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models.schemas import CitationReport, CitationValidateRequest
from app.services.citation_validator import validate_citations

router = APIRouter(prefix="/citations", tags=["citations"])


@router.post("/validate", response_model=CitationReport)
def validate(request: CitationValidateRequest, settings: Settings = Depends(get_settings)) -> CitationReport:
    threshold = request.support_threshold if request.support_threshold is not None else settings.citation_support_threshold
    report = validate_citations(request.answer, request.sources, support_threshold=threshold)
    return CitationReport(
        valid=report.valid,
        cited_sources=report.cited_sources,
        coverage=report.coverage,
        issues=report.issues,
        per_citation=report.per_citation,
    )
