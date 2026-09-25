import pandas as pd
import os
import re
import unicodedata
from sklearn.model_selection import train_test_split

def remove_accents(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def clean_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = remove_accents(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def process_dataset(filepath: str, output_path: str):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep='\t', quoting=3, dtype=str)
    
    # Apply cleaning and rename to match Member 2's contract
    if 'business_name' in df.columns:
        df['business_name_norm'] = df['business_name'].apply(clean_text)
        df = df.drop(columns=['business_name'])
        
    if 'business_address' in df.columns:
        df['business_address_norm'] = df['business_address'].apply(clean_text)
        df = df.drop(columns=['business_address'])
        
    if 'country' in df.columns:
        df['country_norm'] = df['country'].fillna("").astype(str).str.lower().str.strip()
        df = df.drop(columns=['country'])
        
    df.to_csv(output_path, sep='\t', index=False)
    return df

def prepare_data(data_dir: str, output_dir: str):
    """
    Loads, cleans, and saves the train data and a validation split.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_s1 = process_dataset(os.path.join(data_dir, "train_source1.tsv"), os.path.join(output_dir, "cleaned_train_source1.tsv"))
    process_dataset(os.path.join(data_dir, "train_source2.tsv"), os.path.join(output_dir, "cleaned_train_source2.tsv"))
    process_dataset(os.path.join(data_dir, "train_source3.tsv"), os.path.join(output_dir, "cleaned_train_source3.tsv"))
    
    print("Creating validation split...")
    # Using source1_entity_id to match actual ID name
    if 'source1_entity_id' in df_s1.columns:
        id_col = 'source1_entity_id'
    elif 'entity_id' in df_s1.columns:
        id_col = 'entity_id'
    else:
        id_col = df_s1.columns[0]
        
    train_entities, val_entities = train_test_split(df_s1[id_col], test_size=0.2, random_state=42)
    
    train_entities.to_csv(os.path.join(output_dir, "train_split_s1.csv"), index=False)
    val_entities.to_csv(os.path.join(output_dir, "val_split_s1.csv"), index=False)
    
    # Copy GT over
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    if os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path, sep='\t', quoting=3, dtype=str)
        df_gt.to_csv(os.path.join(output_dir, "train_ground_truth.tsv"), sep='\t', index=False)
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../student_resource/dataset/train", help="Path to raw training data")
    parser.add_argument("--output_dir", default="../../data", help="Path to save normalized data")
    args = parser.parse_args()
    
    prepare_data(args.data_dir, args.output_dir)
