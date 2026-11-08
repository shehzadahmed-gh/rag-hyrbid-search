from __future__ import annotations

from app.models.schemas import CitationReport as CitationReportSchema
from app.models.schemas import GenerateResponse, SourceRef
from app.services.citation_validator import validate_citations
from app.services.hybrid_search import HybridSearchService
from app.services.llm_provider import LLMProvider
from app.services.support_checkers import SupportChecker


class GenerationService:
    def __init__(
        self,
        search_service: HybridSearchService,
        support_checker: SupportChecker | None = None,
        support_threshold: float | None = None,
    ):
        self._search = search_service
        self._checker = support_checker
        self._support_threshold = support_threshold

    def answer(self, query: str, llm: LLMProvider, top_k: int = 5) -> GenerateResponse:
        hits, _stage_stats = self._search.search(query, top_k=top_k)

        sources = [
            SourceRef(
                id=i + 1,
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                source=h.source,
                chunk_index=h.chunk_index,
                text=h.text,
            )
            for i, h in enumerate(hits)
        ]

        if not sources:
            answer_text = "No ingested documents matched this query, so there is nothing to ground an answer in."
            report = CitationReportSchema(valid=True, cited_sources=[], coverage=0.0, issues=[], per_citation=[])
            return GenerateResponse(
                query=query, answer=answer_text, provider_used=llm.name, sources=[], citation_report=report
            )

        answer_text = llm.generate_grounded_answer(query, sources)

        report = validate_citations(
            answer_text, sources, checker=self._checker, support_threshold=self._support_threshold
        )
        report_schema = CitationReportSchema(
            valid=report.valid,
            cited_sources=report.cited_sources,
            coverage=report.coverage,
            issues=report.issues,
            per_citation=report.per_citation,
        )

        return GenerateResponse(
            query=query,
            answer=answer_text,
            provider_used=llm.name,
            sources=sources,
            citation_report=report_schema,
        )
