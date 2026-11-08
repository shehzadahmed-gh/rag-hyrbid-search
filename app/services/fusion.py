"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009)."""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    """
    Combine several best-first ranked lists of (doc_id, raw_score) into a
    single fused score per doc_id, using rank position rather than raw
    scores. This sidesteps the fact that BM25 and cosine-similarity scores
    live on completely different, incomparable scales - RRF only needs each
    input list to be correctly *ordered*, not numerically comparable to the
    others.

        score(d) = sum over lists L of  1 / (k + rank_L(d))

    A larger `k` flattens the influence of top ranks (less winner-take-all);
    k=60 is the value used in the original RRF paper and is a reasonable
    default for most corpora.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, (doc_id, _raw_score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position)
    return fused
