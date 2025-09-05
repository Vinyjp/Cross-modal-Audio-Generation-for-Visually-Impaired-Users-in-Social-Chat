import numpy as np

# 加载数据
vectors = np.load("output/image_vectors.npy")
paths = np.load("output/image_paths.npy", allow_pickle=True)

# 打印每张图片的前10维向量
for i, (path, vec) in enumerate(zip(paths, vectors)):
    print(f"\n📌 图片 {i+1}/{len(paths)}: {path}")
    print("前10维向量值:", " ".join([f"{x:.6f}" for x in vec[:10]]))