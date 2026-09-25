from typing import List, Dict, Tuple
import json

def calculate_recall(
    generated_candidates: Dict[str, List[str]], 
    ground_truth: Dict[str, List[str]]
) -> Tuple[float, float, int]:
    """
    Calculate Pair Recall and Full-Entity Recall.
    
    Pair Recall = (Total true S1->S2/S3 links found) / (Total true S1->S2/S3 links)
    Full-Entity Recall = (S1 entities where ALL true matches retrieved) / (S1 entities having >=1 true match)
    """
    total_true_pairs = 0
    found_true_pairs = 0
    
    total_entities_with_matches = 0
    entities_fully_recalled = 0
    
    missed_pairs = []
    
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
            
        total_entities_with_matches += 1
        true_matches_set = set(true_matches)
        total_true_pairs += len(true_matches_set)
        
        cands = set(generated_candidates.get(s1_id, []))
        
        found_matches = true_matches_set & cands
        found_true_pairs += len(found_matches)
        
        if len(found_matches) == len(true_matches_set):
            entities_fully_recalled += 1
        else:
            missed = true_matches_set - cands
            for m in missed:
                missed_pairs.append((s1_id, m))
                
    pair_recall = found_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    full_entity_recall = entities_fully_recalled / total_entities_with_matches if total_entities_with_matches > 0 else 0.0
    
    return pair_recall, full_entity_recall, missed_pairs

def write_metrics(metrics_dict: dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)
