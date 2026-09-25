import argparse

def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Inference")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the test TSV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final TSV outputs")
    args = parser.parse_args()

    print(f"Starting inference pipeline on data in: {args.data_dir}")
    print("1. Preprocessing...")
    print("2. Blocking (generating candidate_pairs.tsv)...")
    print("3. Feature Extraction...")
    print("4. Model Prediction...")
    print(f"5. Saving matching_results.tsv to: {args.output_dir}")

if __name__ == "__main__":
    main()
