import os
import glob
import requests
import numpy as np
import faiss
import json
from pathlib import Path

# ========== 参数配置 ==========
AUDIO_DIR = "data/audio"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIM = 768
API_KEY = "sk-ivjnxwgjseyqhnreysclrlvxkjvmgrctwhaopvdoqcnzctok"  # 你的 API key
INDEX_SAVE_PATH = "filename_vector.index"
METADATA_PATH = "filename_vector_metadata.json"

# ========== 向量提取函数（与你的原代码一致） ==========
def get_embedding(text, model="Qwen/Qwen3-Embedding-8B", dimensions=768):
    url = "https://api.siliconflow.cn/v1/embeddings"
    input_text = text
    payload = {
        "model": model,
        "input": input_text,
        "dimensions": dimensions
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    vector = response.json()['data'][0]['embedding']
    return np.array(vector, dtype=np.float32)

audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.*"))

index = faiss.read_index(INDEX_SAVE_PATH)

embedding = get_embedding("happy", EMBEDDING_MODEL, EMBEDDING_DIM).reshape(1, -1)

distances, indices = index.search(embedding, 10)
for j, i in enumerate(indices[0]):
    path = audio_files[i]
    score = float(distances[0][j])
    print(f"{path} (相似度: {score:.4f})")

