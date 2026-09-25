import pytest
import os
import subprocess
import pandas as pd
from unittest import mock
import sys

# In test_pipeline.py, we run the actual blocking.py script using a subprocess.
# However, to avoid downloading BGE-M3 during testing, we can use unittest.mock
# inside a temporary test runner, or just mock the SentenceTransformer import.

def test_pipeline_smoke(tmp_path):
    """
    Run an end-to-end smoke test of the pipeline using the 10-row synthetic dataset.
    We mock the SentenceTransformer to prevent downloading the actual model.
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'blocking.py')
    s1_path = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_source1.tsv')
    s2_path = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_source2.tsv')
    s3_path = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_source3.tsv')
    gt_path = os.path.join(os.path.dirname(__file__), 'data', 'expected_matches.tsv')
    
    out_dir = str(tmp_path / "data" / "candidates")
    
    # We will run the python script with env variables pointing to our test data
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = out_dir
    
    import importlib.util
    import sys
    
    # Pre-emptively mock the environment so config.py loads it correctly
    with mock.patch.dict(os.environ, {"OUTPUT_ROOT": out_dir}):
        spec = importlib.util.spec_from_file_location("blocking_main", script_path)
        blocking_main = importlib.util.module_from_spec(spec)
        sys.modules["blocking_main"] = blocking_main
        spec.loader.exec_module(blocking_main)
    
    with mock.patch.object(sys, 'argv', ['blocking.py', '--s1', s1_path, '--s2', s2_path, '--s3', s3_path, '--ground_truth', gt_path]):
        with mock.patch('blocking_main.SentenceTransformer') as MockModel:
            # Setup mock model behavior
            instance = MockModel.return_value
            # We must return normalized 1024D vectors
            import numpy as np
            def mock_encode(texts, **kwargs):
                vecs = []
                for text in texts:
                    np.random.seed(abs(hash(text)) % (2**32))
                    vecs.append(np.random.rand(1024).astype('float32'))
                vecs = np.array(vecs)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / norms
            instance.encode.side_effect = mock_encode
            
            # Execute
            blocking_main.main()
            
    # Verify artifacts were generated
    assert os.path.exists(os.path.join(out_dir, 'candidate_pairs.tsv'))
    assert os.path.exists(os.path.join(out_dir, 'blocking_metrics.json'))
    assert os.path.exists(os.path.join(out_dir, 'missed_true_matches.tsv'))
    
    # Verify TSV output
    df = pd.read_csv(os.path.join(out_dir, 'candidate_pairs.tsv'), sep='\t')
    assert 'source1_entity_id' in df.columns
    assert 'candidate_entity_ids' in df.columns
    assert len(df) == 5  # S1 has 5 rows
