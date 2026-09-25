# Amazon ML Challenge 2026 - Pipeline Automation

PYTHON = python
DATA_DIR = ../../student_resource/dataset
OUTPUT_DIR = ../../output
INTERMEDIATE_DIR = ../../data

.PHONY: setup data-prep dummy block features train infer validate all

# 1. Install requirements
setup:
	pip install -r requirements.txt

# 2. (Member 1) Clean data and create validation split
data-prep:
	$(PYTHON) src/preprocess.py --data_dir $(DATA_DIR)/train --output_dir $(INTERMEDIATE_DIR)

# 3. (Member 2) Generate Candidates (Blocking)
block:
	$(PYTHON) src/blocking.py --data_dir $(INTERMEDIATE_DIR) --output_dir $(INTERMEDIATE_DIR)

# 4. (Member 3) Extract Features
features:
	$(PYTHON) src/features.py --data_dir $(INTERMEDIATE_DIR) --output_dir $(INTERMEDIATE_DIR)

# 5. (Member 4) Train Model and Tune Threshold
train:
	$(PYTHON) src/train.py --data_dir $(INTERMEDIATE_DIR) --model_dir models/

# 6. Run Inference on Test Set
infer:
	$(PYTHON) src/inference.py --data_dir $(DATA_DIR)/test --output_dir $(OUTPUT_DIR)

# 7. Validate Submission format
validate:
	$(PYTHON) ../../student_resource/utils/validate_submission.py \
		--matching $(OUTPUT_DIR)/matching_results.tsv \
		--candidate $(OUTPUT_DIR)/candidate_pairs.tsv \
		--test-dir $(DATA_DIR)/test

# Fast-track dummy testing
dummy:
	$(PYTHON) src/dummy_pipeline.py --test_data_dir $(DATA_DIR)/test --output_dir $(OUTPUT_DIR)
	@echo "Now run 'make validate' to test the dummy output."

# Complete marathon run (v1 end-to-end)
all: data-prep block features train infer validate
