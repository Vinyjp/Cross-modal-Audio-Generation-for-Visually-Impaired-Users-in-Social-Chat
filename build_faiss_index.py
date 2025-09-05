import os
import faiss
import numpy as np

# 1. 加载之前生成的向量
vectors = np.load("output/image_vectors.npy").astype('float32')

# 2. 创建FAISS索引（内积相似度）
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(512)
index.add(vectors)

# 3. 保存索引
os.makedirs("output", exist_ok=True)
faiss.write_index(index, "output/image_index.faiss")
print("✅ FAISS索引构建完成！")
print(f"已添加 {index.ntotal} 个向量")