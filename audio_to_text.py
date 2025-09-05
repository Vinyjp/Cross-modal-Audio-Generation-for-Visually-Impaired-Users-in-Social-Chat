from whisper_caption import WhisperForAudioCaptioning
from transformers import WhisperTokenizer, WhisperFeatureExtractor



# Load model

checkpoint = "MU-NLPC/whisper-large-v2-audio-captioning"
model = WhisperForAudioCaptioning.from_pretrained(checkpoint)
tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)

# Load and preprocess audio
input_file = ("data/audio")
audio, sampling_rate = librosa.load(input_file, sr=feature_extractor.sampling_rate)
features = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Prepare caption style
style_prefix = "clotho > caption: "
style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels

# Generate caption
model.eval()
outputs = model.generate(
    inputs=features.to(model.device),
    forced_ac_decoder_ids=style_prefix_tokens,
    max_length=100,
)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
