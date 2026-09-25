# Amazon ML Challenge 2026: Master Architecture & Handoff Contract

> **LLM Context Note:** This document represents the absolute, frozen architectural ground truth. All pipeline phases, modeling decisions, and evaluation frameworks must strictly adhere to the mathematical and operational guidelines defined below. A fixed threshold like 0.91 must never be hard-coded; it serves solely as an initial operational heuristic. The true classification threshold $\tau^*$ must be determined empirically via a fine-grained grid search.

---

## 1. Mandatory Technology Stack and Hardware Strategy

To eliminate environment divergence across teammates, all development must standardize on the pinned environment below. No team member may introduce alternative string libraries, custom vector databases, or alternative boosting frameworks.

### Python 3.11.x Base Environment
*   `pandas==2.2.2`
*   `numpy==1.26.4`
*   `scipy==1.13.1`
*   `scikit-learn==1.5.0`
*   `jellyfish==1.0.4`
*   `datasketch==1.6.5`
*   `sentence-transformers==3.0.1`
*   `torch==2.3.1`
*   `faiss-cpu==1.8.0.post1` *(Use faiss-gpu if CUDA is present)*
*   `lightgbm==4.3.0`
*   `xgboost==2.0.3`

### Hardware Allocation and Device Orchestration
*   **Deep Learning Acceleration:** The dense embedding pipeline must automatically inspect `torch.cuda.is_available()`. If CUDA is active, execute `BAAI/bge-m3` on the GPU with a batch size of 64. If running on CPU, enforce a reduced batch size of 16 to avoid system thrashing.
*   **Vector Search Memory Footprint:** Embeddings must be cached to disk as raw 32-bit floating-point memory-mapped NumPy matrices (`np.memmap` or `.npy`). Never maintain redundant copies of the dense embedding matrix across multiple processes.
*   **FAISS Device Strategy:** If GPU VRAM allows ($\ge 8\text{ GB}$), build the `faiss.IndexFlatIP` on device via `faiss.StandardGpuResources`. Otherwise, execute search on CPU; IndexFlatIP across 1024-dimensional normalized vectors runs efficiently on modern multi-core processors when using OpenMP threading.

---

## 2. Strict Algorithm and Tool Implementation Contracts

### Member 1: Preprocessing & Splitting
*   **Tools:** `pandas`, `re`, `unicodedata`, `csv`
*   **Upstream:** Raw training TSVs: `train_source[1,2,3].tsv`
*   **Downstream:** Cleaned tables: `cleaned_train_source[1,2,3].tsv` + `val_split_s1.csv`
*   **Acceptance:** Verifies zero dropped rows, no null keys, and correct tab separation. Missing or null textual attributes must be handled deterministically (assigned empty strings) with zero runtime exceptions.

### Member 2: Candidate Generation and Blocking
The candidate generation stage must achieve a strict candidate recall of $\ge 99.5\%$ against the training ground truth. Blocking is frozen only after this recall floor is verified.

*   **Dense Semantic Indexing (FAISS):**
    *   **Model:** `BAAI/bge-m3` loaded via `SentenceTransformer` (567M parameters).
    *   **Input Representation:** Concatenated normalized string: `business_name_clean + " " + business_address_clean`.
    *   **Embedding Properties:** Vector dimension $d = 1024$, L2-normalized (`normalize_embeddings=True`), sequence length truncated to 512 tokens.
    *   **Index Architecture:** `faiss.IndexFlatIP(1024)`.
    *   **Retrieval:** Query using Source 1 vectors to retrieve the top $K = 50$ nearest neighbors from the joint Source 2 and Source 3 vector index.
*   **Sparse Lexical LSH Indexing:**
    *   **Library:** `datasketch` (`MinHash`, `MinHashLSH`).
    *   **Shingling:** Character 3-grams over lowercased, accent-stripped text.
    *   **Parameters:** Permutations $P = 128$; Jaccard threshold $\theta = 0.30$.
*   **Candidate Aggregation:** Compute the union of FAISS and LSH candidates. Filter self-references/invalid IDs. Output to `candidate_pairs.tsv`.

### Member 3: Pairwise Feature Engineering
To ensure Member 3 does not spend time designing custom metrics from scratch, the feature space is pinned to these deterministic implementations:

| Feature Name | Exact Implementation | Mathematical Logic |
| :--- | :--- | :--- |
| `feat_jaro_name` | `jellyfish.jaro_winkler_similarity(s1, s2)` | Standard Jaro-Winkler with prefix scaling factor $p=0.1$. |
| `feat_lev_name` | Custom using `jellyfish.levenshtein_distance` | $1.0 - \frac{\text{lev}(s1, s2)}{\max(\text{len}(s1), \text{len}(s2), 1)}$ |
| `feat_monge_elkan_name` | Custom function over whitespace tokens | $\frac{1}{\Vert T_1 \Vert} \sum_{u \in T_1} \max_{v \in T_2} \text{jaro\_winkler}(u, v)$ |
| `feat_jaccard_token_addr`| Native Python set intersection over union | $\frac{\Vert T_1 \cap T_2 \Vert}{\Vert T_1 \cup T_2 \Vert}$ over whitespace address tokens. |
| `feat_soft_tfidf_addr` | Custom using `TfidfVectorizer` | Soft-TFIDF summing product of TF-IDF weights for pairs where jaro_winkler $\ge 0.85$. |
| `feat_cosine_bge_name` | `numpy.dot(vec1, vec2)` | Dot product over unit-normalized `BAAI/bge-m3` name embeddings. |
| `feat_cosine_bge_addr` | `numpy.dot(vec1, vec2)` | Dot product over unit-normalized `BAAI/bge-m3` address embeddings. |
| `feat_digit_overlap` | `re.findall(r'\d+', text)` | Intersection ratio of numeric digit tokens. |
| `feat_country_match` | Exact string equality | `int(country_s1.strip() == country_target.strip())` |

### Member 4: Supervised Classification & Dynamic Threshold Tuning
Implement both LightGBM and XGBoost, followed by an empirical ensemble and threshold grid search:

*   **Model A (LightGBM):** `n_estimators=1000, learning_rate=0.03, num_leaves=63, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42`.
*   **Model B (XGBoost):** `n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42`.
*   **Model C (Ensemble Blending):**
    $P_{\text{blend}} = w \cdot P_{\text{LGBM}} + (1 - w) \cdot P_{\text{XGB}}$
    *(Evaluate weights $w \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$)*

**Dynamic Macro $F_{0.5}$ Empirical Threshold Sweep:**
1. Collect out-of-fold predictions across `GroupKFold(n_splits=5)` partitioned on `source1_entity_id`.
2. Iterate candidate thresholds $\tau \in [0.500, 0.990]$ in increments of 0.005.
3. For each threshold $\tau$, assign candidate pairs as positive matches if $P_{\text{blend}} \ge \tau$.
4. Compute precision, recall, and instance $F_{0.5}$ for every validation entity.
5. **Singleton Logic:** Assign singletons with an empty prediction an instance score of 1.0; singletons predicted with ANY false match receive 0.0.
6. Average the entity-level scores globally to compute macro $F_{0.5}$. Lock the optimal threshold $\tau^*$.

---

## 3. Automated Submission Acceptance and Anti-Leakage Protocol

Before any submission archive is constructed, execute:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
The pipeline must assert an exit code of 0 (PASS), enforcing:
1.  **Row Parity:** Every `source1_entity_id` appears exactly once.
2.  **Strict Subset Containment:** Matched IDs must exist in `candidate_pairs.tsv`.
3.  **Strict Source Scoping:** Matched identifiers must only contain `S2-` and `S3-` prefixes.
4.  **Delimiter Precision:** All files are strictly tab-separated (`\t`).
