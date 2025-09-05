import librosa
import numpy as np
import os

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)  # 返回20维特征

# 批量处理
audio_dir = "data/audio"
features = []
paths = []
for file in os.listdir(audio_dir):
    if file.endswith(('.mp3', '.wav')):
        path = os.path.join(audio_dir, file)
        feat = extract_audio_features(path)
        features.append(feat)
        paths.append(path)

# 保存结果
os.makedirs("output/audio_vectors", exist_ok=True)
np.save("output/audio_vectors/audio_vectors.npy", np.array(features))
np.save("output/audio_vectors/audio_paths.npy", np.array(paths))
print(f"✅ 已处理 {len(features)} 个音频文件")