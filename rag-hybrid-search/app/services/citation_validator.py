"""
Citation validation - the fourth modular service.

Checks two independent things about a generated answer against the numbered
source chunks it was supposed to be grounded in:

  1. Range validity   - every [n] marker in the answer must refer to a
                         source that was actually retrieved (catches the
                         model inventing a citation number that doesn't
                         exist).
  2. Lexical support   - for each cited sentence, does the cited source
                         actually contain overlapping content? A citation
                         to a real source that doesn't support the specific
                         claim is still a grounding failure.

Support is measured with token-set Jaccard overlap between the cited
sentence and its source chunk. This is intentionally simple and has no
external dependency - it is a heuristic screen for obviously-unsupported
citations, not a semantic entailment model. A production system would layer
an NLI/entailment model or an LLM-as-judge check on top for a stricter
signal; the interface here (`validate_citations`) is where that would slot
in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.chunking import split_sentences

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "for", "and", "or", "but", "with", "as", "by", "at", "from", "that", "this", "it", "its",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _strip_citations(text: str) -> str:
    return _CITATION_PATTERN.sub("", text)


def _support_score(sentence: str, source_text: str) -> float:
    sent_tokens = _tokens(_strip_citations(sentence))
    source_tokens = _tokens(source_text)
    if not sent_tokens or not source_tokens:
        return 0.0
    return len(sent_tokens & source_tokens) / len(sent_tokens)


@dataclass
class CitationReport:
    valid: bool
    cited_sources: list[int]
    coverage: float
    issues: list[str] = field(default_factory=list)
    per_citation: list[dict] = field(default_factory=list)


def validate_citations(answer: str, sources: list, support_threshold: float = 0.15) -> CitationReport:
    """
    `sources` is a list of objects with `.id` (1-indexed, matching the [n]
    markers) and `.text` (the source chunk content).
    """
    valid_ids = {s.id for s in sources}
    source_by_id = {s.id: s for s in sources}

    cited_numbers = sorted({int(n) for n in _CITATION_PATTERN.findall(answer)})
    issues: list[str] = []

    out_of_range = [n for n in cited_numbers if n not in valid_ids]
    if out_of_range:
        issues.append(
            f"Citation marker(s) {out_of_range} do not match any retrieved source "
            f"(valid range is 1-{len(sources)}); this looks like a hallucinated citation."
        )

    if not cited_numbers:
        issues.append("The answer contains no [n] citation markers - none of its claims are traceably grounded.")

    per_citation: list[dict] = []
    for sentence in split_sentences(answer):
        for n in {int(x) for x in _CITATION_PATTERN.findall(sentence)}:
            if n not in valid_ids:
                continue
            source = source_by_id[n]
            score = _support_score(sentence, source.text)
            supported = score >= support_threshold
            per_citation.append(
                {
                    "citation": n,
                    "sentence": sentence,
                    "support_score": round(score, 3),
                    "supported": supported,
                }
            )
            if not supported:
                issues.append(
                    f"Citation [{n}] has low lexical overlap ({score:.2f}) with source [{n}]; "
                    "the claim may not actually be supported by that excerpt."
                )

    coverage = len(set(cited_numbers) & valid_ids) / len(sources) if sources else 0.0

    return CitationReport(
        valid=len(issues) == 0,
        cited_sources=cited_numbers,
        coverage=round(coverage, 3),
        issues=issues,
        per_citation=per_citation,
    )
