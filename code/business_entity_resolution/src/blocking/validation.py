from typing import List, Set, Dict

def validate_integrity(
    final_candidates: Dict[str, List[str]], 
    valid_target_ids: Set[str], 
    mode: str = "train"
) -> bool:
    """
    Strict integrity validation to prevent ID leakage.
    Ensures that candidate IDs are strictly a subset of the target universe.
    """
    for s1_id, candidates in final_candidates.items():
        # Condition 1: No S1 ID can be a candidate
        if s1_id in candidates:
            print(f"[FAIL] Entity {s1_id} contains itself in candidate list.")
            return False
        
        # Condition 2: No duplicates within a single candidate string
        if len(candidates) != len(set(candidates)):
            print(f"[FAIL] Entity {s1_id} contains duplicate candidate IDs.")
            return False
        
        # Condition 3: Candidate IDs ⊆ Target ID Universe
        for c_id in candidates:
            if c_id not in valid_target_ids:
                print(f"[FAIL] Candidate ID {c_id} for {s1_id} does not exist in {mode} target IDs.")
                return False
                
    return True
