# Business Entity Resolution Pipeline

This repository contains the end-to-end machine learning pipeline for the Amazon ML Challenge 2026.

## Structure
- `src/preprocess.py`: Data ingestion, cleaning, and normalization.
- `src/blocking.py`: Candidate generation (TF-IDF, Faiss, Phonetics).
- `src/features.py`: Pairwise similarity feature extraction.
- `src/train.py`: Model training (XGBoost/LightGBM) and threshold tuning.
- `src/inference.py`: End-to-end inference script generating final outputs.

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the raw datasets are placed in the `dataset/` directory at the project root (relative to the final zip structure).

## How to Reproduce

To reproduce the end-to-end pipeline (Data -> Blocking -> Matching -> Output):

```bash
# 1. Run inference (this will call preprocessing, blocking, feature generation, and model prediction)
python src/inference.py --data_dir ../../dataset/test --output_dir ../../output/
```

*(Note: The actual training can be re-run using `python src/train.py` if needed)*
