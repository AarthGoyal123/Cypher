# Member 2: Candidate Generation Action Plan (Production-Grade)

As Member 2, your entire goal is to maximize **Candidate Recall** while minimizing the $O(N^2)$ search space. If the true match is not in your output file, the ML classifier (Member 4) can never predict it.

This plan contains exact contracts, frozen parameters, and mandatory experiments. Do not guess; execute these steps precisely.

---

## Step 1: Hardware & Runtime Contract (Kaggle/Colab)
Your local laptop is for development, Git, editing, and lightweight testing only. You will execute the heavy candidate-generation workload (BGE-M3, FAISS, LSH) inside Kaggle or Google Colab.

**Path Agnostic Architecture:**
Do NOT hardcode Kaggle (`/kaggle/input/...`) or Colab (`/content/...`) paths. Use environment variables:
```python
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "dataset")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "data/candidates")
```

**Automatic Hardware Detection:**
Your script must detect the environment to dynamically adjust constraints and log them to `blocking_metrics.json`:
```python
import torch
import psutil

has_gpu = torch.cuda.is_available()
if has_gpu:
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
else:
    device = "cpu"
    gpu_name = None
    gpu_vram_gib = 0

system_ram_gib = psutil.virtual_memory().total / (1024 ** 3)
```

---

## Step 2: Environment Setup
Do not blindly force PyTorch versions; let Kaggle/Colab handle CUDA stacks. Ensure these libraries are installed:

```bash
pip install sentence-transformers==3.0.1 datasketch==1.6.5 pandas==2.2.2 numpy==1.26.4 psutil
```
*Note: If the runtime provides a compatible GPU-enabled FAISS, use it. Otherwise, `faiss-cpu` is a valid implementation; the bottleneck is retrieval speed, not correctness.*

---

## Step 3: Frozen Text Representation (The Contract)
You must use an explicit, frozen format to concatenate Member 1's cleaned columns. This prevents representation mismatch between queries and targets.

```python
def build_retrieval_text(row):
    return (
        f"business name: {row['business_name_clean']} "
        f"address: {row['business_address_clean']}"
    )
```
*Apply this exact function to Source 1, Source 2, and Source 3 strings.*

---

## Step 4: Implement Semantic Retrieval (FAISS + BGE-M3)
1.  **Initialize Model:** Load `SentenceTransformer('BAAI/bge-m3')`.
2.  **Encode Targets (CHUNKED):** **DO NOT encode the entire target dataset at once, regardless of its actual row count.** Process in chunks: Load Target Chunk $\rightarrow$ Encode $\rightarrow$ `FAISS.add()` $\rightarrow$ Delete temporary arrays.
3.  **Batch Sizing:** Initial: GPU `128` | CPU `32`. If OOM, halve the batch size ($128 \rightarrow 64 \rightarrow 32 \rightarrow 16$) until stable.
4.  **Enforce Normalization (Crucial):**
    ```python
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    assert embeddings.shape[1] == 1024
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)
    ```
5.  **Calculate FAISS Memory:** `IndexFlatIP` stores every vector in RAM. Estimate size and log to metrics:
    `estimated_faiss_gb = (target_count * 1024 * 4) / (1024 ** 3)`
6.  **Persist ID Mapping:** NEVER assume `index_position == entity_id`. Maintain a mapping array: `target_ids = [S2-001, S2-002, S3-001...]`.
7.  **Build FAISS Index:** Initialize `faiss.IndexFlatIP(1024)`. Add embeddings.
8.  **Query FAISS:** Encode Source 1 strings. Retrieve top $K$ results.

---

## Step 5: Implement Lexical Retrieval (MinHash LSH)
1.  **Tokenization Contract:** Apply this exact character 3-gram function to the frozen retrieval text:
```python
def char_ngrams(text, n=3):
    return {text[i:i+n] for i in range(len(text) - n + 1)}
```
2.  **API Contract:** Use exact `datasketch` APIs.
```python
from datasketch import MinHash, MinHashLSH
m = MinHash(num_perm=128)
lsh = MinHashLSH(threshold=0.30, num_perm=128)
```
3.  **Memory-Safe Generation:** **DO NOT build one giant Python list of MinHash objects in RAM (`all_minhashes = []` is strictly prohibited).** Process in chunks: Text $\rightarrow$ 3-Grams $\rightarrow$ MinHash $\rightarrow$ LSH insertion $\rightarrow$ Discard local MinHash reference.

---

## Step 6: Aggregation & Deterministic Deduplication
Member 2 owns the final hybrid candidate set. Do not write separate files for Member 3 to merge.
```text
FAISS K-NN  AND  MinHash LSH
     │                │
     ▼                ▼
 FAISS IDs         LSH IDs
     └───────┬────────┘
             ▼
         SET UNION
             ▼
        SORT + DEDUP
             ▼
    candidate_pairs.tsv
```
```python
candidate_ids = sorted(set(faiss_ids) | set(lsh_ids))
```

---

## Step 7: Mandatory Experimentation Matrix
Do NOT blindly assume $K=50$ or $\theta=0.30$ are optimal. Run this exact experiment matrix on the training set:

| Experiment | FAISS $K$ | LSH $\theta$ | Purpose |
| :--- | :--- | :--- | :--- |
| A | 25 | OFF | semantic baseline |
| B | 50 | OFF | semantic baseline |
| C | 75 | OFF | semantic baseline |
| D | 100 | OFF | semantic baseline |
| E | 150 | OFF | semantic baseline |
| F | OFF | 0.20 | lexical baseline |
| G | OFF | 0.25 | lexical baseline |
| H | OFF | 0.30 | lexical baseline |
| I | OFF | 0.35 | lexical baseline |
| J | selected K | selected $\theta$ | hybrid union |

Record: pair recall, full-entity recall, avg/median/max candidates, and runtime.

---

## Step 8: Handoff & Delivery Artifacts
You must deliver an auditable artifacts directory.

**Directory Structure:**
```text
data/candidates/
├── candidate_pairs.tsv       <-- Exact official schema
├── blocking_metrics.json     <-- Scientific audit
└── missed_true_matches.tsv   <-- Diagnostic file
```

**JSON Metrics Contract (`blocking_metrics.json`):**
```json
{
    "runtime": {
        "platform": null,
        "device": null,
        "gpu": null,
        "gpu_vram_gib": null,
        "system_ram_gib": null
    },
    "faiss_k": null,
    "lsh_threshold": null,
    "minhash_num_perm": 128,
    "character_ngram": 3,
    "embedding_batch_size": null,
    "estimated_faiss_gib": null,
    "faiss_pair_recall": null,
    "lsh_pair_recall": null,
    "hybrid_pair_recall": null,
    "full_entity_recall": null,
    "source1_count": null,
    "target_count": null,
    "total_candidate_pairs": null,
    "average_candidates_per_source1": null,
    "median_candidates_per_source1": null,
    "max_candidates_per_source1": null,
    "runtime_seconds": null
}
```

---

## Member 2 — Definition of Done
**MEMBER 2 IS NOT DONE UNTIL ALL CONDITIONS PASS.**

**BLOCKING**
- [ ] FAISS semantic retrieval implemented
- [ ] MinHash LSH lexical retrieval implemented
- [ ] Both independently benchmarked via Experiment Matrix
- [ ] Hybrid union implemented

**SEMANTIC**
- [ ] 1024-dimensional embeddings explicitly asserted
- [ ] Normalized embeddings explicitly asserted
- [ ] FAISS memory (GiB) estimated and recorded
- [ ] `index_position` $\rightarrow$ `entity_id` array persisted
- [ ] FAISS chunking executed

**LEXICAL**
- [ ] `char_ngrams(n=3)` applied exclusively to frozen retrieval text
- [ ] `all_minhashes = []` pattern successfully avoided (no OOM)

**RECALL**
- [ ] Pair recall & Full-entity recall calculated
- [ ] **TARGET:** Hybrid pair recall $\ge 99.5\%$ (MUST INVESTIGATE if $< 99.5\%$)
- [ ] `missed_true_matches.tsv` diagnostic generated

**INTEGRITY & LEAKAGE (CRITICAL)**
- [ ] Train ID validation and Test ID validation are strictly separated
- [ ] No Train/Test IDs are mixed
- [ ] For train: `candidate_entity_ids ⊆ train_source2.entity_id ∪ train_source3.entity_id`
- [ ] For test: `candidate_entity_ids ⊆ test_source2.entity_id ∪ test_source3.entity_id`
- [ ] No `S1-` candidates in candidate list
- [ ] No duplicates within a single candidate string
- [ ] Deterministic ordering (`sorted()`)
- [ ] Every `S1` entity from the source file is represented exactly once

**HANDOFF**
- [ ] `candidate_pairs.tsv` generated with exact official schema
- [ ] `blocking_metrics.json` populated with actual run values
