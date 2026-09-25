import pandas as pd

def build_retrieval_text(row: pd.Series) -> str:
    """
    Frozen contract for text concatenation.
    This guarantees zero representation drift between queries and targets.
    """
    name = str(row.get('business_name_norm', '')) if pd.notna(row.get('business_name_norm')) else ""
    address = str(row.get('business_address_norm', '')) if pd.notna(row.get('business_address_norm')) else ""
    
    return f"business name: {name} address: {address}"

def char_ngrams(text: str, n: int = 3) -> set:
    """
    Apply character n-grams to the frozen retrieval text.
    Do not apply independently to raw fields.
    """
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}
