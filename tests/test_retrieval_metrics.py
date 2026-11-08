import math

import pytest

from app.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k


def test_recall_counts_relevant_documents_in_top_k():
    assert recall_at_k(["a", "x", "b", "y"], {"a", "b", "c", "d"}, k=3) == 0.5
    assert recall_at_k(["a"], set(), k=10) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank_at_k(["x", "y", "a", "b"], {"a", "b"}, k=10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_ndcg_is_one_for_ideal_ranking_and_discounts_lower_hits():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=10) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a"], {"a", "b"}, k=10) == pytest.approx(expected)


def test_ndcg_ideal_is_capped_at_k():
    # 20 relevant documents but only 2 slots: two hits is a perfect score.
    relevant = {str(i) for i in range(20)}
    assert ndcg_at_k(["0", "1"], relevant, k=2) == pytest.approx(1.0)
