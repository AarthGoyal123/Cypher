# Amazon ML Challenge 2026: Deep Dataset Context & Schema

> **LLM Context Note:** This document provides full structural details, noise profiles, and volume metrics of the training dataset. If you are an LLM building features or tuning hyperparameters for this project, you must base your assumptions on the statistics and schemas provided below.

---

## 1. Overall Dataset Topology

The dataset simulates a massive, real-world Entity Resolution scenario. It is divided into 3 independent data sources containing business identities.
*   **Source 1 (`train_source1.tsv`):** The deduplicated "Source of Truth" or reference source. **Size: ~200 MB.** Every row here is a unique real-world entity.
*   **Source 2 (`train_source2.tsv`):** A highly noisy secondary source. **Size: ~466 MB.** Contains multiple noisy duplicates that map back to Source 1.
*   **Source 3 (`train_source3.tsv`):** A tertiary noisy source, similar in structure and volume to Source 2.

**Goal:** Given a record in Source 1, find *all* instances of that record residing in Source 2 and Source 3.

---

## 2. File Schema & Data Types

ALL files (including ground truth and your final submission) use a **Strict Tab-Separated (`\t`)** format. 
*Why?* The data contains raw, unescaped commas inside the `business_address` and `matched_entity_ids` columns. Standard CSV parsing will fail.

### The Source Files (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`)
| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `entity_id` | String | `S1-925783039` | Unique identifier. Prefix denotes the source (`S1-`, `S2-`, `S3-`). |
| `business_name` | String | `Orelee's Barbershop` | Unstructured business name. Highly susceptible to typos. |
| `business_address`| String | `1795 Westchester Drive, High Point, NC` | Unstructured address. May contain suite numbers, states, landmarks. |
| `country` | String | `US` | Ground truth country code (`US`, `India`, and in the test set: `France`). |

### The Ground Truth File (`train_ground_truth.tsv`)
| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `source1_entity_id`| String | `S1-00001` | The reference entity. |
| `matched_entity_ids`| String | `S2-00047,S2-00193,S3-00812` | Comma-separated list of matches. **Empty string** if no matches exist (Singleton). |

---

## 3. Data Anomalies & Noise Profiles

The ML pipeline (specifically the feature engineering phase) must be explicitly programmed to handle the following noise profiles observed in the dataset.

### A. Business Name Noise
1.  **Legal Suffix Inconsistencies:** `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`.
2.  **Punctuation Discrepancies:** `&` vs `and`, `B+ Retail Inc` vs `B Plus Retail`.
3.  **Typos & Transpositions:** `Prime Money` vs `Primu Mony`. 
4.  **Unicode Characters:** The dataset contains non-ASCII characters (causing `cp1252` encoding failures in native Windows). NLTK and pandas must explicitly use `utf-8` encoding.

### B. Address Noise
1.  **Component Reordering:** `Unit APARTMENT G, 2100 Cameron Drive` vs `2100 Cameron Drive, Unit G`.
2.  **Abbreviations:** `Rd` vs `Road`, `St` vs `Street`, `Bldg` vs `Building`.
3.  **Missing Geographical Data:** Indian addresses frequently omit State or PIN Codes.
4.  **Landmark References:** Addresses may contain unparseable human references like `Near SBI ATM` or `Opposite City Center`.

### C. The Country Column (Crucial Rule)
*   **The Trap:** The training dataset only contains `US` and `India`. 
*   **The Rule:** The test dataset contains a third label (`France`) which does **not** appear in training.
*   **Actionable Insight for LLMs:** Do **NOT** one-hot encode the `country` column. Treat it as a categorical string or use binary equality checking (`country_match = 1` if exact match). Hard-coding to `{US, India}` will instantly break the pipeline during evaluation.

---

## 4. Implementation Guidelines for Feature Engineering
Based on this dataset profile, LLMs building features should prioritize:
1.  **Jaro-Winkler Similarity** on `business_name` (heavily rewards matching prefixes, great for typos).
2.  **Soft-TFIDF** on `business_address` (great for re-ordered components and missing ZIP codes).
3.  **BGE-M3 Dense Embeddings** (The 8192 context window is necessary to absorb the entire address string + business name without truncating). 
