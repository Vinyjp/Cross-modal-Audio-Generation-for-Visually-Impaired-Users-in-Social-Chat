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

# ========== 初始化 FAISS 索引 和 元数据 ==========
index = faiss.IndexFlatIP(EMBEDDING_DIM)
metadata = []

# ========== 遍历音频文件，提取文件名文本并向量化 ==========
audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.*"))

for file_path in audio_files:
    filename = Path(file_path).stem  # e.g. happy_001
    print(f"\n🎧 当前文件: {filename}")

    try:
        # 仅使用文件名作为文本输入
        embedding = get_embedding(filename, EMBEDDING_MODEL, EMBEDDING_DIM)
        index.add(np.array([embedding]))

        metadata.append({
            "filename": filename,
            "embedding": embedding.tolist()  # 方便保存查看
        })

    except Exception as e:
        print(f"❌ 处理 {filename} 时出错：{e}")

# ========== 保存 FAISS 索引和 JSON 元数据 ==========
faiss.write_index(index, INDEX_SAVE_PATH)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\n✅ 所有文件名向量化完成，已保存索引和元数据。")