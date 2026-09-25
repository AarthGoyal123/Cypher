import pytest
from blocking.metrics import calculate_recall

def test_calculate_recall():
    ground_truth = {
        "S1-001": ["S2-001", "S3-001"],
        "S1-002": ["S2-002"],
        "S1-003": []  # No match
    }
    
    # S1-001 missed S3-001
    # S1-002 got all its true matches (plus an extra false positive, which doesn't hurt recall)
    # S1-003 is irrelevant for recall computation
    candidates = {
        "S1-001": ["S2-001"],            
        "S1-002": ["S2-002", "S3-005"],  
        "S1-003": ["S2-005"]             
    }
    
    pair_recall, full_entity, missed = calculate_recall(candidates, ground_truth)
    
    # Total True Pairs = 3. Found Pairs = 2.
    assert pair_recall == 2 / 3
    
    # Total Entities With Match = 2. Fully recalled = 1 (S1-002).
    assert full_entity == 1 / 2
    
    # Check that S3-001 was accurately recorded as a miss
    assert ("S1-001", "S3-001") in missed
