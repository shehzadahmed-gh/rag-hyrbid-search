from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_support_checker
from app.models.schemas import CitationReport, CitationValidateRequest
from app.services.citation_validator import validate_citations
from app.services.support_checkers import SupportChecker

router = APIRouter(prefix="/citations", tags=["citations"])


@router.post("/validate", response_model=CitationReport)
def validate(
    request: CitationValidateRequest,
    settings: Settings = Depends(get_settings),
    checker: SupportChecker = Depends(get_support_checker),
) -> CitationReport:
    threshold = request.support_threshold
    if threshold is None:
        threshold = settings.citation_support_threshold
    report = validate_citations(request.answer, request.sources, checker=checker, support_threshold=threshold)
    return CitationReport(
        valid=report.valid,
        cited_sources=report.cited_sources,
        coverage=report.coverage,
        issues=report.issues,
        per_citation=report.per_citation,
    )
