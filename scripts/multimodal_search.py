import faiss
import numpy as np
import json


class MultimodalSearcher:
    def __init__(self):
        # 加载资源
        self.index = faiss.read_index("output/unified/multimodal_index.faiss")
        with open("output/unified/metadata.json") as f:
            self.meta = json.load(f)

        # 加载路径数据
        self.image_paths = np.load("output/image_paths.npy")
        self.text_labels = np.load("output/text_vectors/emotion_labels.npy")
        self.audio_paths = np.load("output/audio_vectors/audio_paths.npy")

    def search_by_image(self, image_path, top_k=3):
        """核心功能：图片→匹配图片→对应音频"""
        from clip_feature_extractor import extract_features

        # 1. 提取查询图片向量
        query_vec = extract_features(image_path)
        query_vec = np.load("output/unified/image_vecs_20d.npy")[0]  # 示例用第一张图片

        # 2. 在统一索引中搜索
        d, i = self.index.search(query_vec.reshape(1, -1), top_k)

        # 3. 结果解析
        results = []
        for idx, score in zip(i[0], d[0]):
            if idx <= self.meta["image_range"][1]:
                # 图片结果
                img_path = self.image_paths[idx]
                audio_idx = np.where(self.audio_paths == img_path.replace(".jpg", ".mp3"))[0]
                if len(audio_idx) > 0:
                    results.append({
                        "type": "image",
                        "path": img_path,
                        "audio": self.audio_paths[audio_idx[0]],
                        "score": float(score)
                    })
        return results


# 使用示例
if __name__ == "__main__":
    searcher = MultimodalSearcher()
    results = searcher.search_by_image("test.jpg")  # 替换为你的测试图片路径

    print("🏆 匹配结果：")
    for res in results:
        print(f"图片: {res['path']}")
        print(f"音频: {res['audio']}")
        print(f"相似度: {res['score']:.4f}\n")