import faiss
import numpy as np
from clip_feature_extractor import extract_features
import os
import glob
import requests
import numpy as np
import faiss

AUDIO_DIR = "data/audio"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIM = 768
API_KEY = "sk-ivjnxwgjseyqhnreysclrlvxkjvmgrctwhaopvdoqcnzctok"  # 你的 API key
INDEX_SAVE_PATH = "filename_vector.index"

index_image = faiss.read_index("output/image_index.faiss")
image_paths = np.load("output/image_paths.npy")

audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.*"))
index_audio = faiss.read_index(INDEX_SAVE_PATH)

# 2. 定义搜索函数
def search_similar(query_image_path, index, top_k=3):
    query_vec = extract_features(query_image_path).astype('float32')
    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec.reshape(1, -1), top_k)
    return [(image_paths[i], float(distances[0][j])) for j, i in enumerate(indices[0])]

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
    # 3. 测试检索
if __name__ == "__main__":
    results = search_similar("data/images/test.png", index_image, 1) # 换成你的测试图片路径
    for path, score in results:
        # 处理path的文件名
        embedding = get_embedding(path, EMBEDDING_MODEL, EMBEDDING_DIM).reshape(1, -1)
        distances, indices = index_audio.search(embedding, 1)
        for j, i in enumerate(indices[0]):
            path_audio = audio_files[i]
            score_audio = float(distances[0][j])
            print("最相似的音频：")
            print(f"{path_audio} (相似度: {score_audio:.4f})")