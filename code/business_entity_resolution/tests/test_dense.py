import pytest
import numpy as np
import faiss
from blocking.dense import encode_in_chunks, build_faiss_index, query_faiss

class MockEncoder:
    """Mocks BGE-M3 to return 1024-dimensional normalized vectors"""
    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vecs = []
        for text in texts:
            # deterministic pseudorandom vector for each string
            np.random.seed(abs(hash(text)) % (2**32))
            vecs.append(np.random.rand(1024).astype('float32'))
        vecs = np.array(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

def test_encode_in_chunks_and_build():
    model = MockEncoder()
    texts = ["business name: a address: b", "business name: c address: d", "business name: e address: f"]
    
    # Test Chunk Equivalence (Extremely important for Kaggle/Colab reproducibility)
    emb_chunk2 = encode_in_chunks(model, texts, chunk_size=2)
    emb_chunk5 = encode_in_chunks(model, texts, chunk_size=5)
    
    assert np.allclose(emb_chunk2, emb_chunk5), "Chunking logic altered the embedding output!"
    
    # Assertions from the contract
    assert emb_chunk2.shape == (3, 1024)
    assert np.allclose(np.linalg.norm(emb_chunk2, axis=1), 1.0, atol=1e-3)
    
    # Build FAISS
    index = build_faiss_index(emb_chunk2)
    assert index.ntotal == 3

def test_faiss_id_mapping():
    model = MockEncoder()
    target_texts = ["T1", "T2"]
    target_ids = ["S2-001", "S3-001"]
    
    emb = encode_in_chunks(model, target_texts)
    index = build_faiss_index(emb)
    
    # Query with exact same embeddings should return index 0 for 0, index 1 for 1
    results = query_faiss(index, emb, target_ids, k=1)
    assert results[0] == ["S2-001"]
    assert results[1] == ["S3-001"]
