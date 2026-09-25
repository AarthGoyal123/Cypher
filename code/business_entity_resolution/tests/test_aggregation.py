import pytest
from blocking.aggregation import aggregate_candidates

def test_aggregate_candidates():
    faiss_res = {0: ["S2-001", "S2-003"], 1: []}
    lsh_res = {0: ["S2-002", "S2-003"], 1: ["S3-001"]}
    s1_ids = ["S1-001", "S1-002"]
    
    res = aggregate_candidates(faiss_res, lsh_res, s1_ids)
    
    # S1-001 should have the union of FAISS and LSH, deterministically sorted
    assert res["S1-001"] == sorted(["S2-001", "S2-002", "S2-003"])
    
    # S1-002 should only have S3-001
    assert res["S1-002"] == ["S3-001"]

def test_aggregate_removes_s1():
    # Test deliberate injection of an S1 ID into the candidates
    faiss_res = {0: ["S1-001", "S2-001"]}
    lsh_res = {0: []}
    s1_ids = ["S1-001"]
    
    res = aggregate_candidates(faiss_res, lsh_res, s1_ids)
    
    assert "S1-001" not in res["S1-001"]
    assert res["S1-001"] == ["S2-001"]
