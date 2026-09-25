import faiss
import numpy as np
import gc
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_BATCH_SIZE

def encode_in_chunks(model: SentenceTransformer, texts: List[str], chunk_size: int = 50000) -> np.ndarray:
    """Encode texts in memory-safe chunks to prevent OOM."""
    all_embeddings = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        emb = model.encode(
            chunk, 
            batch_size=EMBEDDING_BATCH_SIZE, 
            normalize_embeddings=True, 
            convert_to_numpy=True, 
            show_progress_bar=True
        )
        all_embeddings.append(emb)
        # Force garbage collection to keep RAM stable
        gc.collect()
        
    embeddings_matrix = np.vstack(all_embeddings)
    
    # Mathematical assertion from the contract
    assert embeddings_matrix.shape[1] == 1024, "Embeddings must be 1024 dimensions"
    assert np.allclose(np.linalg.norm(embeddings_matrix, axis=1), 1.0, atol=1e-3), "Embeddings must be L2 normalized"
    
    return embeddings_matrix

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build the FAISS IndexFlatIP. Embeddings must be normalized."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def query_faiss(index: faiss.IndexFlatIP, query_embeddings: np.ndarray, target_ids: List[str], k: int = 50) -> dict:
    """Query the FAISS index and return a mapping of query_idx -> list of target IDs."""
    distances, indices = index.search(query_embeddings, k)
    
    results = {}
    for i, idx_row in enumerate(indices):
        results[i] = [target_ids[idx] for idx in idx_row if idx != -1]
    return results
