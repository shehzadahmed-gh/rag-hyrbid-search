from app.services.fusion import reciprocal_rank_fusion


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("b", 5.0), ("a", 4.0), ("d", 3.0)]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    # "a" is rank 1 in dense and rank 2 in bm25; "b" is rank 2 in dense and rank 1 in bm25
    # -> both should outscore "c" and "d", which only appear in one list
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
    assert fused["a"] == fused["b"]  # symmetric ranks (1+2 vs 2+1) should tie


def test_rrf_document_only_in_one_list_still_scored():
    fused = reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]], k=60)
    assert set(fused.keys()) == {"a", "b"}
    assert fused["a"] == fused["b"]  # both are rank 1 in their respective single list


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == {}


def test_rrf_smaller_k_increases_top_rank_dominance():
    ranked = [("first", 1.0), ("second", 1.0)]
    fused_small_k = reciprocal_rank_fusion([ranked], k=1)
    fused_large_k = reciprocal_rank_fusion([ranked], k=1000)
    gap_small_k = fused_small_k["first"] - fused_small_k["second"]
    gap_large_k = fused_large_k["first"] - fused_large_k["second"]
    assert gap_small_k > gap_large_k
