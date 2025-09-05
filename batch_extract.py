import os
import numpy as np
from clip_feature_extractor import extract_features # 复用第一周的CLIP工具函数
from tqdm import tqdm # 进度条工具

def batch_extract_images(image_dir="data/images", output_dir="output"):
    """
    批量提取图片特征向量
    :param image_dir: 图片目录路径
    :param output_dir: 输出目录路径
    """
    # 0. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 收集所有图片路径
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f" 错误：{image_dir} 中没有找到图片！")
        return

    print(f" 找到 {len(image_paths)} 张图片，开始提取特征...")

    # 2. 批量提取特征
    vectors = []
    valid_paths = []
    for path in tqdm(image_paths, desc="处理进度"):
        vec = extract_features(path)
        if vec is not None:
            vectors.append(vec)
            valid_paths.append(path)

    # 3. 保存结果
    vectors_np = np.array(vectors)
    np.save(os.path.join(output_dir, "image_vectors.npy"), vectors_np)
    np.save(os.path.join(output_dir, "image_paths.npy"), np.array(valid_paths))

    print(f"✅ 完成！已保存 {len(vectors)} 个向量到 {output_dir}/")

if __name__ == "__main__":
    batch_extract_images() # 直接运行