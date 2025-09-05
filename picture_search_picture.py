import faiss
import numpy as np
from sympy.polys.polyconfig import query

from clip_feature_extractor import extract_features

# 1. 加载索引和图片路径

index = faiss.read_index("output/image_index.faiss")
image_paths = np.load("output/image_paths.npy")

# 2. 定义搜索函数
def search_similar(query_image_path, top_k=3):

    query_vec = extract_features(query_image_path).astype('float32')

    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec.reshape(1, -1), top_k)
    return [(image_paths[i], float(distances[0][j])) for j, i in enumerate(indices[0])]

    # 3. 测试检索
if __name__ == "__main__":
    results = search_similar("data/sad.png") # 换成你的测试图片路径
    print("最相似的图片：")
    for path, score in results:
        print(f"{path} (相似度: {score:.4f})")