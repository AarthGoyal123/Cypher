from typing import List, Dict

def aggregate_candidates(
    faiss_results: Dict[int, List[str]], 
    lsh_results: Dict[int, List[str]], 
    s1_ids: List[str]
) -> Dict[str, List[str]]:
    """
    Computes the deterministic union of FAISS and LSH candidates.
    Removes any self-references (S1 IDs) and returns a sorted candidate list per S1 entity.
    """
    final_candidates = {}
    
    for i, s1_id in enumerate(s1_ids):
        # Default to empty set if index missing
        f_cands = set(faiss_results.get(i, []))
        l_cands = set(lsh_results.get(i, []))
        
        # Exact deterministic union
        union_cands = f_cands | l_cands
        
        # Self-match prevention
        if s1_id in union_cands:
            union_cands.remove(s1_id)
            
        # Deterministic sorting guarantees reproducible candidate output
        final_candidates[s1_id] = sorted(list(union_cands))
        
    return final_candidates
