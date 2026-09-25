from datasketch import MinHash, MinHashLSH
from typing import List, Dict
import gc
from .config import MINHASH_NUM_PERM, LSH_THRESHOLD, NGRAM_SIZE
from .retrieval_text import char_ngrams

def build_lsh_index(texts: List[str], target_ids: List[str], chunk_size: int = 50000) -> MinHashLSH:
    """
    Builds the MinHashLSH index in chunks. 
    Strictly forbids accumulating all MinHash objects in RAM.
    """
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    
    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        chunk_ids = target_ids[i:i + chunk_size]
        
        for text, tid in zip(chunk_texts, chunk_ids):
            m = MinHash(num_perm=MINHASH_NUM_PERM)
            for ngram in char_ngrams(text, NGRAM_SIZE):
                m.update(ngram.encode('utf8'))
            lsh.insert(tid, m)
            # Local MinHash reference 'm' is overwritten/released immediately
        
        # Explicit garbage collection to maintain low overhead
        gc.collect()
        
    return lsh

def query_lsh(lsh: MinHashLSH, texts: List[str]) -> Dict[int, List[str]]:
    """Query the LSH index for candidate matches."""
    results = {}
    for i, text in enumerate(texts):
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        for ngram in char_ngrams(text, NGRAM_SIZE):
            m.update(ngram.encode('utf8'))
        candidates = lsh.query(m)
        results[i] = candidates
    return results
