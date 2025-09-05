import faiss
import numpy as np
import os
from clip_feature_extractor import extract_features

# 1. 收集所有图片路径
image_dir = "data/images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# 2. 提取所有图片特征
vectors = []
valid_paths = []
for path in image_paths:
    vec = extract_features(path)
    if vec is not None:
        vectors.append(vec)
        valid_paths.append(path)

# 3. 构建FAISS索引
vectors = np.array(vectors).astype('float32')
index = faiss.IndexFlatIP(512) # 内积相似度
index.add(vectors)

# 4. 保存索引和路径映射
faiss.write_index(index, "image_index.faiss")
np.save("image_paths.npy", np.array(valid_paths))
print(f"已构建索引，包含 {len(valid_paths)} 张图片")