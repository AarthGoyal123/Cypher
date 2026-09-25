# Business Entity Resolution: High-Precision Architecture (98-99% F0.5 Target)

To achieve a 98-99% score on an F0.5 metric (which penalizes false positives heavily) without relying on external APIs, we must build a **hybrid Machine Learning pipeline** that combines high-recall blocking with highly precise supervised feature matching.

Here is the state-of-the-art methodology designed specifically for this challenge:

## 1. Aggressive Data Preprocessing & Normalization
Dirty data is the biggest enemy of precision. Before any matching occurs, we must normalize the fields:
*   **Text Cleaning:** Lowercasing, removing special characters (e.g., `&` to `and`), and standardizing whitespace.
*   **Entity Suffix Expansion:** Mapping abbreviations to full words (`Corp` -> `Corporation`, `Pvt` -> `Private`, `Ltd` -> `Limited`).
*   **Address Normalization:** Standardizing road types (`Rd` -> `Road`, `St` -> `Street`) and handling country-specific patterns (US vs. India formats).

## 2. High-Recall Candidate Generation (Blocking)
We cannot compare every Source 1 record to every Source 2/Source 3 record ($O(N^2)$). We need to generate a candidate pool (the `candidate_pairs.tsv`) that captures nearly 100% of true matches while ignoring obvious non-matches.
*   **TF-IDF & Cosine Similarity:** Create character n-gram TF-IDF vectors for `business_name` and `business_address`. Find nearest neighbors using optimized similarity search (like FAISS or scikit-learn).
*   **Phonetic Blocking:** Create blocking keys using algorithms like Soundex or Double Metaphone to group names that sound similar but are spelled differently.
*   **MinHash LSH:** Extremely fast fuzzy string matching to cast a wide net.

## 3. Advanced Pairwise Feature Engineering
For every candidate pair generated in Step 2, we compute a rich set of similarity features. This is where the model learns the nuances.
*   **Lexical Distances:** 
    *   *Jaro-Winkler:* Excellent for capturing typos in names.
    *   *Levenshtein Distance:* General edit distance.
    *   *Monge-Elkan:* Great for multi-word strings where words might be reordered.
*   **Token-based Similarities:** Jaccard index, Soft-TFIDF.
*   **Semantic Embeddings (Deep Learning):** We will use a lightweight pre-trained Transformer (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to embed names and addresses into dense vectors and calculate cosine similarity. This helps the model understand that "IBM" and "International Business Machines" are semantically close, even if lexically different, keeping us within the "No External DBs" and "<8B parameters" constraints.

## 4. Supervised Classification & Threshold Tuning
Instead of hard-coded rules, we feed the similarity features from Step 3 into a powerful Gradient Boosting classifier.
*   **Model:** XGBoost, LightGBM, or CatBoost. These handle tabular feature data exceptionally well and train incredibly fast.
*   **The Secret to High F0.5:** The F0.5 score values Precision twice as much as Recall. A standard ML model predicts a match if the probability is > 0.5. We will use a hold-out validation set to plot a **Precision-Recall Curve** and find the exact probability threshold (e.g., 0.85 or 0.92) that maximizes the F0.5 score. We will be very conservative before declaring a match.

## 5. Post-Processing & Transitive Closure
*   Ensure that if A matches B, and B matches C, then A matches C (if logically sound).
*   **Singletons:** Correctly predicting "no match" is worth a full 1.0. Our high confidence threshold will naturally result in many empty `matched_entity_ids`, successfully handling singletons.
