import faiss
import numpy as np

# 加载统一后的向量
image_vecs = np.load("output/unified/image_vecs_20d.npy")
text_vecs = np.load("output/unified/text_vecs_20d.npy")
audio_vecs = np.load("output/unified/audio_vecs_20d.npy")

# 合并并构建索引
all_vecs = np.vstack([image_vecs, text_vecs, audio_vecs])
index = faiss.IndexFlatIP(20)
index.add(all_vecs)
faiss.write_index(index, "output/unified/multimodal_index.faiss")

# 生成元数据
metadata = {
    "image_range": [0, len(image_vecs)-1],
    "text_range": [len(image_vecs), len(image_vecs)+len(text_vecs)-1],
    "audio_range": [len(image_vecs)+len(text_vecs), len(all_vecs)-1]
}
import json
with open("output/unified/metadata.json", "w") as f:
    json.dump(metadata, f)

print("✅ 统一索引构建完成！")