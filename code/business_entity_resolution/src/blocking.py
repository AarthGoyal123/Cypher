# Main orchestrator for Member 2
import argparse
import time
import os
import pandas as pd
import numpy as np

from blocking.config import (
    HAS_GPU, DEVICE, GPU_NAME, GPU_VRAM_GIB, SYSTEM_RAM_GIB,
    DATA_ROOT, OUTPUT_ROOT, FAISS_K, LSH_THRESHOLD
)
from blocking.retrieval_text import build_retrieval_text
from blocking.dense import encode_in_chunks, build_faiss_index, query_faiss
from blocking.lexical import build_lsh_index, query_lsh
from blocking.aggregation import aggregate_candidates
from blocking.validation import validate_integrity
from blocking.metrics import calculate_recall, write_metrics
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge: Member 2 Blocking Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode determines which dataset IDs to validate against')
    parser.add_argument('--s1', type=str, required=True, help='Path to Source 1 TSV')
    parser.add_argument('--s2', type=str, required=True, help='Path to Source 2 TSV')
    parser.add_argument('--s3', type=str, required=True, help='Path to Source 3 TSV')
    parser.add_argument('--ground_truth', type=str, help='Path to Ground Truth TSV (required for train mode)')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    start_time = time.time()
    
    print("=== Member 2 Blocking Pipeline ===")
    print(f"Device: {DEVICE}")
    if HAS_GPU:
        print(f"GPU: {GPU_NAME} ({GPU_VRAM_GIB:.1f} GiB VRAM)")
    print(f"System RAM: {SYSTEM_RAM_GIB:.1f} GiB\n")
    
    # 1. Load Data
    print("Loading datasets...")
    s1_df = pd.read_csv(args.s1, sep='\t', quoting=3, dtype=str)
    s2_df = pd.read_csv(args.s2, sep='\t', quoting=3, dtype=str)
    s3_df = pd.read_csv(args.s3, sep='\t', quoting=3, dtype=str)
    
    # 2. Build Frozen Retrieval Text
    print("Building frozen retrieval text...")
    s1_texts = s1_df.apply(build_retrieval_text, axis=1).tolist()
    s2_texts = s2_df.apply(build_retrieval_text, axis=1).tolist()
    s3_texts = s3_df.apply(build_retrieval_text, axis=1).tolist()
    
    s1_ids = s1_df['entity_id'].tolist() if 'entity_id' in s1_df.columns else s1_df.iloc[:, 0].tolist()
    target_ids = (s2_df['entity_id'].tolist() if 'entity_id' in s2_df.columns else s2_df.iloc[:, 0].tolist()) + \
                 (s3_df['entity_id'].tolist() if 'entity_id' in s3_df.columns else s3_df.iloc[:, 0].tolist())
    target_texts = s2_texts + s3_texts
    target_ids_set = set(target_ids)
    
    # 3. Dense FAISS Pipeline
    print("Running BGE-M3 + FAISS pipeline...")
    model = SentenceTransformer('BAAI/bge-m3')
    target_embeddings = encode_in_chunks(model, target_texts)
    faiss_index = build_faiss_index(target_embeddings)
    query_embeddings = encode_in_chunks(model, s1_texts)
    faiss_results = query_faiss(faiss_index, query_embeddings, target_ids, k=FAISS_K)
    
    # 4. Lexical MinHash LSH Pipeline
    print("Running MinHash LSH pipeline...")
    lsh_index = build_lsh_index(target_texts, target_ids)
    lsh_results = query_lsh(lsh_index, s1_texts)
    
    # 5. Union & Deduplication
    print("Aggregating candidates...")
    final_candidates = aggregate_candidates(faiss_results, lsh_results, s1_ids)
    
    # 6. Integrity Validation
    print("Validating integrity...")
    is_valid = validate_integrity(final_candidates, target_ids_set, mode=args.mode)
    if not is_valid:
        raise ValueError("Integrity validation failed! Output aborted to prevent pipeline corruption.")
    
    # 7. Metrics (if train)
    metrics_log = {
        "runtime": {
            "platform": "auto",
            "device": DEVICE,
            "gpu": GPU_NAME,
            "gpu_vram_gib": GPU_VRAM_GIB,
            "system_ram_gib": SYSTEM_RAM_GIB
        },
        "faiss_k": FAISS_K,
        "lsh_threshold": LSH_THRESHOLD,
        "estimated_faiss_gib": (len(target_ids) * 1024 * 4) / (1024 ** 3),
        "source1_count": len(s1_ids),
        "target_count": len(target_ids)
    }
    
    # Calculate Candidate Volume
    cand_lengths = [len(cands) for cands in final_candidates.values()]
    metrics_log["total_candidate_pairs"] = sum(cand_lengths)
    metrics_log["average_candidates_per_source1"] = float(np.mean(cand_lengths)) if cand_lengths else 0
    metrics_log["median_candidates_per_source1"] = float(np.median(cand_lengths)) if cand_lengths else 0
    metrics_log["max_candidates_per_source1"] = int(np.max(cand_lengths)) if cand_lengths else 0
    
    if args.mode == 'train' and args.ground_truth:
        print("Calculating Recall...")
        gt_df = pd.read_csv(args.ground_truth, sep='\t', quoting=3, dtype=str)
        # Parse match list format string like "['S2-xxx', 'S3-yyy']"
        import ast
        gt_dict = {}
        for _, row in gt_df.iterrows():
            s1 = row['source1_entity_id']
            matches = ast.literal_eval(row['matched_entity_ids'])
            gt_dict[s1] = matches
            
        pair_recall, full_entity_recall, missed = calculate_recall(final_candidates, gt_dict)
        metrics_log["hybrid_pair_recall"] = pair_recall
        metrics_log["full_entity_recall"] = full_entity_recall
        print(f"Pair Recall: {pair_recall:.4f}")
        print(f"Full Entity Recall: {full_entity_recall:.4f}")
        
        # Write missed matches
        missed_df = pd.DataFrame(missed, columns=['source1_entity_id', 'missed_target_id'])
        missed_df.to_csv(os.path.join(OUTPUT_ROOT, 'missed_true_matches.tsv'), sep='\t', index=False)
        
    metrics_log["runtime_seconds"] = time.time() - start_time
    write_metrics(metrics_log, os.path.join(OUTPUT_ROOT, 'blocking_metrics.json'))
    
    # 8. Export Handoff Artifact
    print("Writing candidate_pairs.tsv...")
    out_rows = []
    for s1, cands in final_candidates.items():
        out_rows.append({"source1_entity_id": s1, "candidate_entity_ids": ",".join(cands)})
    pd.DataFrame(out_rows).to_csv(os.path.join(OUTPUT_ROOT, 'candidate_pairs.tsv'), sep='\t', index=False)
    print("Done!")

if __name__ == "__main__":
    main()
