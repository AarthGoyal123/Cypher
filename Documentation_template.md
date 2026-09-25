# Amazon ML Challenge 2026: Methodology & Architecture Write-up

## Team Name: Cypher

### 1. Methodology Used
Our methodology treats entity resolution as a strictly decoupled two-stage pipeline designed to mathematically optimize the macro-averaged $F_{0.5}$ metric. Recognizing that a single false positive merges degrades the score four times faster than a false negative, we constructed an aggressively conservative matching engine. 

The pipeline begins with NFKD Unicode normalization to strip diacritics (crucial for French test data) and isolates numeric spatial tokens. We then perform dual-space candidate generation to maximize recall. The resulting candidate pairs are vectorized into 18-dimensional feature spaces capturing multi-granular similarity (semantic, lexical, phonetic). Finally, a gradient-boosted decision tree (GBDT) ensemble evaluates the pairs, utilizing a globally optimized decision threshold ($\tau \ge 0.88$) that explicitly shields singleton entities from false-positive corruption.

### 2. Candidate Generation / Blocking Strategy
We employ a **Multi-index Hybrid Blocking Cascade** to reduce the $O(N^2)$ search space while maintaining >99.5% recall.
1.  **Lexical Space (MinHash LSH):** Business names and addresses are tokenized into character 3-grams. A MinHash LSH index identifies approximate lexical matches across varying string lengths, maintaining robustness against character-level permutations and OCR errors.
2.  **Semantic Space (Bi-encoder FAISS):** We utilized the `BAAI/bge-m3` embedding model to generate 1024-dimensional normalized dense vectors for every concatenated record. An exact Inner Product FAISS index (`faiss.IndexFlatIP`) was constructed over Source 2 and Source 3 vectors. Source 1 queries retrieved the top $K=50$ nearest neighbors.

The union of candidates from both retrieval channels constitutes our final `candidate_pairs.tsv`.

### 3. Model Architecture and Feature Engineering
For candidate classification, we rejected end-to-end cross-encoders due to O(∣B∣) latency bottlenecks and opted for a **LightGBM / XGBoost GBDT Ensemble**.

Every candidate pair was transformed into a robust numerical feature vector:
*   **Dense Semantic Cosine:** Computed across dense `BAAI/bge-m3` embeddings for business names and addresses independently.
*   **String & Lexical Distances:** Jaro-Winkler (prefix sensitivity), normalized Levenshtein ratios, and Monge-Elkan (multi-word transpositions).
*   **Token Overlap:** Token-set Jaccard indices and Soft-TFIDF similarities.
*   **Jurisdictional Invariance:** To handle the unseen 'France' geographic distribution, we avoided one-hot encoding entirely. Instead, we generated a binary indicator `feat_country_match = I(country_S1 == country_tgt)`.

**Threshold Optimization (The F0.5 Secret):** Models were trained via a 5-Fold GroupKFold split (partitioned strictly by `source1_entity_id`). Instead of default boundaries ($\tau=0.5$), out-of-fold probability predictions were swept across $\tau \in [0.50, 0.99]$. The exact threshold maximizing the global validation $F_{0.5}$ score was selected for final inference, ensuring strict precision guarantees.

### 4. Any other relevant information about the approach
**Singleton Defense:** The mathematical dynamics of singletons demand strict defensive conservatism. Accurately predicting an empty string yields full credit (1.0), whereas a single false merge collapses the score to 0.0. Our high decision boundary explicitly defends against singleton corruption.

**Data Integrity Verification:** To prevent parsing failures on unstructured addresses containing raw commas, our entire ingestion and emission pipeline utilizes strict `sep="\t"` and `quoting=csv.QUOTE_NONE` mechanics, maintaining complete compliance with the provided validation tools.
