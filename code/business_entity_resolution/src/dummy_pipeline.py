import pandas as pd
import random
import os

def calculate_f05(precision: float, recall: float) -> float:
    """Calculate F_0.5 score."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def create_dummy_submission(test_data_dir: str, output_dir: str):
    """Creates a random candidate set and random matching result for testing the pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test source 1 to get entity IDs
    df_s1 = pd.read_csv(os.path.join(test_data_dir, "test_source1.tsv"), sep='\t', quoting=3)
    df_s2 = pd.read_csv(os.path.join(test_data_dir, "test_source2.tsv"), sep='\t', quoting=3)
    
    # Get a sample of S2 IDs to randomly assign
    s2_ids = df_s2['entity_id'].tolist()
    
    candidate_records = []
    matching_records = []
    
    for s1_id in df_s1['entity_id']:
        # Randomly pick 1-5 candidates
        k_candidates = random.randint(1, 5)
        candidates = random.sample(s2_ids, k=min(k_candidates, len(s2_ids)))
        candidate_str = ",".join(candidates) if candidates else ""
        candidate_records.append({"source1_entity_id": s1_id, "candidate_entity_ids": candidate_str})
        
        # Randomly pick 0-3 matches from the candidates
        k_matches = random.randint(0, min(3, len(candidates)))
        matches = random.sample(candidates, k=k_matches)
        match_str = ",".join(matches) if matches else ""
        matching_records.append({"source1_entity_id": s1_id, "matched_entity_ids": match_str})
        
    df_candidates = pd.DataFrame(candidate_records)
    df_matches = pd.DataFrame(matching_records)
    
    # Save files
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    
    df_candidates.to_csv(cand_path, sep='\t', index=False)
    df_matches.to_csv(match_path, sep='\t', index=False)
    
    print(f"Dummy candidate_pairs.tsv saved to {cand_path}")
    print(f"Dummy matching_results.tsv saved to {match_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", default="../../student_resource/dataset/test", help="Path to test data")
    parser.add_argument("--output_dir", default="../../output", help="Path to save dummy submission")
    args = parser.parse_args()
    
    create_dummy_submission(args.test_data_dir, args.output_dir)
