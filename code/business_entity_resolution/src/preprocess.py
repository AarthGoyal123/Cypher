import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load the tsv file correctly."""
    # The problem statement mentions reading with explicit \t separator
    return pd.read_csv(filepath, sep='\t')

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean business names and addresses.
    TODO: Implement Member 1's logic here.
    """
    df_cleaned = df.copy()
    # Add cleaning logic (lowercasing, suffix expansion, etc.)
    return df_cleaned

if __name__ == "__main__":
    print("Preprocessing module ready.")
