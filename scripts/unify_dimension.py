import numpy as np
from sklearn.decomposition import PCA

# 加载所有向量
image_vecs = np.load("output/image_vectors.npy")  # 512维
text_vecs = np.load("output/text_vectors/emotion_vectors.npy")  # 384维
audio_vecs = np.load("output/audio_vectors/audio_vectors.npy")  # 20维

# 统一到20维（以音频维度为准）
pca = PCA(n_components=20)
image_vecs = pca.fit_transform(image_vecs.astype('float32'))
text_vecs = pca.fit_transform(text_vecs.astype('float32'))
audio_vecs = audio_vecs.astype('float32')

# 保存
os.makedirs("output/unified", exist_ok=True)
np.save("output/unified/image_vecs_20d.npy", image_vecs)
np.save("output/unified/text_vecs_20d.npy", text_vecs)
np.save("output/unified/audio_vecs_20d.npy", audio_vecs)