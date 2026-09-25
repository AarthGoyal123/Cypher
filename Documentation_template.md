# Amazon ML Challenge 2026: Methodology Document

## Team Name: [Enter Team Name]

### 1. Methodology Used
[Provide a high-level overview of your end-to-end pipeline. Explain how you connected data preprocessing, blocking, feature engineering, and the final classification model.]

### 2. Candidate Generation / Blocking Strategy
[Detail how you generated `candidate_pairs.tsv`. Explain the algorithms used (e.g., TF-IDF, MinHash, Soundex) and how you achieved high recall while keeping the number of candidate pairs manageable.]

### 3. Model Architecture and Feature Engineering
[Describe the machine learning model used for final matching (e.g., XGBoost, LightGBM, neural networks). List the key pairwise features you engineered (e.g., Jaro-Winkler, cosine similarity of embeddings) and explain how you tuned the decision threshold for the F0.5 metric.]

### 4. Any other relevant information about the approach
[Include any specific handling for singletons, transitivity logic, address normalization tricks, or challenges faced during the implementation.]
