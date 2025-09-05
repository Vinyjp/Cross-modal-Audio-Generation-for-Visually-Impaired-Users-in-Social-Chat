# -*- coding: utf-8 -*-
"""
Audio Search API (by text)
POST /api/search_audio
Body: {"text": "<basename_without_ext>", "topk": 1}
Return: {"ok": true, "audio_path": "<best match audio absolute/relative path>", "score": <float>}
Prereqs:
  - FAISS audio index: filename_vector.index
  - AUDIO_DIR with the same order used when building the index
  - SiliconFlow embeddings API key (set API_KEY below or via env SILICONFLOW_API_KEY)
"""
from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import os, glob, numpy as np, faiss, requests, base64

AUDIO_DIR = "data/audio"
INDEX_SAVE_PATH = "filename_vector.index"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIM = 768
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-ivjnxwgjseyqhnreysclrlvxkjvmgrctwhaopvdoqcnzctok")  # 替换或用环境变量
SILICONFLOW_EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"

app = Flask(__name__)

_index = None
_audio_files = None

def _list_audio_files():
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.*")))
    if not files:
        raise FileNotFoundError(f"未在 {AUDIO_DIR} 下找到音频文件")
    return files

def _load_audio_index():
    global _index, _audio_files
    if _index is None:
        if not os.path.exists(INDEX_SAVE_PATH):
            raise FileNotFoundError(f"未找到音频索引文件：{INDEX_SAVE_PATH}")
        _index = faiss.read_index(INDEX_SAVE_PATH)
        _audio_files = _list_audio_files()
    return _index, _audio_files

def _get_text_embedding(text: str):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text, "dimensions": EMBEDDING_DIM}
    resp = requests.post(SILICONFLOW_EMBED_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    vec = data["data"][0]["embedding"]
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    # 如果索引为余弦相似/内积流程，建议归一化
    faiss.normalize_L2(arr)
    return arr

def _guess_mime_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".mp3":
        return "audio/mpeg"
    elif ext == ".wav":
        return "audio/wav"
    elif ext == ".ogg":
        return "audio/ogg"
    elif ext == ".flac":
        return "audio/flac"
    elif ext == ".m4a":
        return "audio/mp4"
    elif ext == ".aac":
        return "audio/aac"
    else:
        return "application/octet-stream"

@app.route("/api/search_audio", methods=["POST"])
def search_audio():
    if not request.is_json:
        return jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    topk = int(data.get("topk", 1))
    if not text:
        return jsonify({"ok": False, "error": "Missing field: text"}), 400
    if topk < 1:
        topk = 1

    try:
        index, audio_files = _load_audio_index()
        q = _get_text_embedding(text)
        distances, indices = index.search(q, topk)
        audio_path_list=[]
        scores=[]
        audio_base64_list=[]
        audio_mime_list=[]
        for i in range(topk):
            idx = int(indices[0][i])
            score = float(distances[0][i])
            audio_path = audio_files[idx]
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_mime = _guess_mime_type(audio_path)
            audio_path_list.append(audio_path)
            scores.append(score)
            audio_base64_list.append(audio_base64)
            audio_mime_list.append(audio_mime)
        return jsonify({"ok": True, "audio_path": audio_path_list, "score": scores, "audio_base64": audio_base64_list, "audio_mime": audio_mime_list}), 200
    except requests.HTTPError as he:
        return jsonify({"ok": False, "error": f"Embedding API error: {he.response.text}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code or 500
    return jsonify({"ok": False, "error": str(e)}), code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7002, debug=True)
