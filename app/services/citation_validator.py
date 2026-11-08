"""Checks a generated answer's [n] citations against the sources it was given.

Two separate checks:

1. Every [n] must refer to a source that was actually retrieved. A number
   outside that range is an invented citation.
2. Each cited claim must be supported by the source it cites. The answer is
   split into sentences, and each sentence into claims: the text before a
   marker (or a run of markers like [1][2]) is the claim for those sources.
   "A is true [1] and B is true [2]." is checked as "A is true" against
   source 1 and "and B is true" against source 2.

How "supported" is judged depends on the SupportChecker (see
support_checkers.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.chunking import split_sentences
from app.services.support_checkers import LexicalSupportChecker, SupportChecker

_CITATION = re.compile(r"\[(\d+)\]")
_CITATION_RUN = re.compile(r"(?:\[\d+\][\s,;]*)+")


@dataclass
class CitationReport:
    valid: bool
    cited_sources: list[int]
    coverage: float
    issues: list[str] = field(default_factory=list)
    per_citation: list[dict] = field(default_factory=list)


def split_claims(sentence: str) -> list[tuple[str, list[int]]]:
    """[(claim_text, [source ids]), ...] for one sentence, in order."""
    claims: list[tuple[str, list[int]]] = []
    position = 0
    for run in _CITATION_RUN.finditer(sentence):
        text = sentence[position : run.start()].strip(" ,;")
        ids = [int(n) for n in _CITATION.findall(run.group())]
        claims.append((text, ids))
        position = run.end()
    return claims


def validate_citations(
    answer: str,
    sources: list,
    checker: SupportChecker | None = None,
    support_threshold: float | None = None,
) -> CitationReport:
    """`sources` are objects with `.id` (1-indexed, matching [n]) and `.text`."""
    checker = checker or LexicalSupportChecker()
    threshold = checker.default_threshold if support_threshold is None else support_threshold
    source_by_id = {s.id: s for s in sources}

    cited_numbers = sorted({int(n) for n in _CITATION.findall(answer)})
    issues: list[str] = []

    out_of_range = [n for n in cited_numbers if n not in source_by_id]
    if out_of_range:
        issues.append(
            f"Citation marker(s) {out_of_range} do not match any retrieved source "
            f"(valid range is 1-{len(sources)}); this looks like a hallucinated citation."
        )
    if not cited_numbers:
        issues.append("The answer contains no [n] citation markers, so none of its claims can be traced to a source.")

    to_score: list[tuple[str, int]] = []
    for sentence in split_sentences(answer):
        for claim, ids in split_claims(sentence):
            for n in dict.fromkeys(ids):  # de-duplicate, keep order
                if n in source_by_id:
                    to_score.append((claim, n))

    scores = checker.score_many([(claim, source_by_id[n].text) for claim, n in to_score])
    per_citation: list[dict] = []
    for (claim, n), score in zip(to_score, scores, strict=True):
        supported = score >= threshold
        per_citation.append({"citation": n, "claim": claim, "support_score": round(score, 3), "supported": supported})
        if not supported:
            issues.append(
                f"Citation [{n}] scored {score:.2f} with the {checker.name} checker (threshold {threshold}); "
                f"source [{n}] may not support: {claim!r}"
            )

    coverage = len(set(cited_numbers) & source_by_id.keys()) / len(sources) if sources else 0.0
    return CitationReport(
        valid=not issues,
        cited_sources=cited_numbers,
        coverage=round(coverage, 3),
        issues=issues,
        per_citation=per_citation,
    )
