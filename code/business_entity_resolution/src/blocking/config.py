import os
import torch
import psutil

DATA_ROOT = os.environ.get("DATA_ROOT", "dataset")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "data/candidates")

HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else None
GPU_VRAM_GIB = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if HAS_GPU else 0
SYSTEM_RAM_GIB = psutil.virtual_memory().total / (1024 ** 3)

# Initial baseline configurations
FAISS_K = 50
LSH_THRESHOLD = 0.30
MINHASH_NUM_PERM = 128
NGRAM_SIZE = 3
EMBEDDING_BATCH_SIZE = 128 if HAS_GPU else 32
TARGET_CHUNK_SIZE = 50000
