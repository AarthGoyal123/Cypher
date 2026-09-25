import pytest
from blocking.lexical import build_lsh_index, query_lsh

def test_lsh_pipeline():
    target_texts = ["apple", "google"]
    target_ids = ["S2-001", "S2-002"]
    
    lsh = build_lsh_index(target_texts, target_ids, chunk_size=1)
    
    # The LSH should contain the exact IDs
    assert lsh.is_empty() is False
    
    # Querying the exact same text should yield the ID
    query_texts = ["apple"]
    results = query_lsh(lsh, query_texts)
    
    assert "S2-001" in results[0]
    assert "S2-002" not in results[0]
