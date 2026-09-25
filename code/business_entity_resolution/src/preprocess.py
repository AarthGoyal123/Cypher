import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """Load the tsv file correctly."""
    # Read with explicit \t separator
    return pd.read_csv(filepath, sep='\t', quoting=3) # quoting=3 is QUOTE_NONE to avoid issue with random quotes

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean business names and addresses.
    """
    df_cleaned = df.copy()
    
    # Check if 'business_name' and 'business_address' exist before applying string ops
    for col in ['business_name', 'business_address']:
        if col in df_cleaned.columns:
            # Lowercase, fill NA, and strip
            df_cleaned[col] = df_cleaned[col].fillna("").astype(str).str.lower().str.strip()
            # Remove special characters
            df_cleaned[col] = df_cleaned[col].apply(lambda x: re.sub(r'[^a-z0-9\s]', ' ', x))
            # Standardize multiple spaces to single space
            df_cleaned[col] = df_cleaned[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            
    return df_cleaned

def prepare_data(data_dir: str, output_dir: str):
    """
    Loads, cleans, and saves the train data and a validation split.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading raw training data...")
    df_s1 = load_data(os.path.join(data_dir, "train_source1.tsv"))
    df_s2 = load_data(os.path.join(data_dir, "train_source2.tsv"))
    df_s3 = load_data(os.path.join(data_dir, "train_source3.tsv"))
    df_gt = load_data(os.path.join(data_dir, "train_ground_truth.tsv"))
    
    print("Cleaning data...")
    df_s1_clean = clean_and_normalize(df_s1)
    df_s2_clean = clean_and_normalize(df_s2)
    df_s3_clean = clean_and_normalize(df_s3)
    
    # Save cleaned files
    print("Saving normalized tables...")
    df_s1_clean.to_csv(os.path.join(output_dir, "cleaned_train_source1.tsv"), sep='\t', index=False)
    df_s2_clean.to_csv(os.path.join(output_dir, "cleaned_train_source2.tsv"), sep='\t', index=False)
    df_s3_clean.to_csv(os.path.join(output_dir, "cleaned_train_source3.tsv"), sep='\t', index=False)
    
    # Validation split on Source 1 entities (80/20)
    print("Creating validation split...")
    train_entities, val_entities = train_test_split(df_s1_clean['entity_id'], test_size=0.2, random_state=42)
    
    # Save splits
    train_entities.to_csv(os.path.join(output_dir, "train_split_s1.csv"), index=False)
    val_entities.to_csv(os.path.join(output_dir, "val_split_s1.csv"), index=False)
    df_gt.to_csv(os.path.join(output_dir, "train_ground_truth.tsv"), sep='\t', index=False) # copy gt over
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../student_resource/dataset/train", help="Path to raw training data")
    parser.add_argument("--output_dir", default="../../data", help="Path to save normalized data")
    args = parser.parse_args()
    
    prepare_data(args.data_dir, args.output_dir)
