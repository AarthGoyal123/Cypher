import pytest
from blocking.validation import validate_integrity

def test_validate_integrity_success():
    valid_ids = {"S2-001", "S2-002"}
    assert validate_integrity({"S1-001": ["S2-001"]}, valid_ids)

def test_validate_integrity_s1_inclusion():
    valid_ids = {"S2-001"}
    # Fail: S1 ID mistakenly leaked into candidates
    assert not validate_integrity({"S1-001": ["S1-001", "S2-001"]}, valid_ids)

def test_validate_integrity_unknown_target():
    valid_ids = {"S2-001"}
    # Fail: Candidate ID does not exist in target universe
    assert not validate_integrity({"S1-001": ["S3-999"]}, valid_ids)

def test_validate_integrity_duplicates():
    valid_ids = {"S2-001"}
    # Fail: Duplicate IDs present in candidate string
    assert not validate_integrity({"S1-001": ["S2-001", "S2-001"]}, valid_ids)
