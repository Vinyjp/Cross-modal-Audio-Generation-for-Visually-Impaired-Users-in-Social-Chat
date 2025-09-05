import clip
import torch
from PIL import Image


# 初始化CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model")

def extract_features(image_path):
    try:
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image_input)
        return features.cpu().numpy().flatten() # 转换为512维向量
    except Exception as e:
        print(f"处理失败: {image_path} - 错误: {e}")
        return None

# 测试单张图片
if __name__ == "__main__":
    features = extract_features("test.jpg") # 替换为你的测试图片路径
    print("特征向量示例:", features[:5], "... 维度:", len(features))