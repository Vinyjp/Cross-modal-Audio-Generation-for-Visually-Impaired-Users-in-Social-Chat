# -*- coding: utf-8 -*-
"""
Image Search API
POST /api/search_image
Body: {"image_base64": "<base64 string>", "topk": 1}
Return: {"ok": true, "name": "<matched image basename without extension>", "score": <float>}
Prereqs:
  - FAISS image index: output/image_index.faiss
  - Image paths npy:   output/image_paths.npy
  - clip_feature_extractor.extract_features(image_path) -> np.ndarray[float32]
"""
from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import os, base64, tempfile
import numpy as np
import faiss
from clip_feature_extractor import extract_features

IMAGE_INDEX_PATH = "output/image_index.faiss"
IMAGE_PATHS_NPY   = "output/image_paths.npy"

app = Flask(__name__)

# Lazy globals
_index = None
_img_paths = None

def _load_image_index():
    global _index, _img_paths
    if _index is None:
        if not os.path.exists(IMAGE_INDEX_PATH):
            raise FileNotFoundError(f"未找到图像索引文件：{IMAGE_INDEX_PATH}")
        if not os.path.exists(IMAGE_PATHS_NPY):
            raise FileNotFoundError(f"未找到图像路径列表：{IMAGE_PATHS_NPY}")
        _index = faiss.read_index(IMAGE_INDEX_PATH)
        _img_paths = np.load(IMAGE_PATHS_NPY, allow_pickle=True)
    return _index, _img_paths

def _decode_base64_to_temp(b64_str: str) -> str:
    raw = base64.b64decode(b64_str)
    # 写临时文件（jpg后缀即可，特征提取只需能被读取）
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return tmp_path

@app.route("/api/search_image", methods=["POST"])
def search_image():
    if not request.is_json:
        return jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    b64 = data.get("image_base64")
    topk = int(data.get("topk", 1))
    if not b64:
        return jsonify({"ok": False, "error": "Missing field: image_base64"}), 400
    if topk < 1:
        topk = 1

    index, img_paths = _load_image_index()

    # 解码并提取特征
    tmp_path = None
    try:
        tmp_path = _decode_base64_to_temp(b64)
        vec = extract_features(tmp_path).astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        distances, indices = index.search(vec, topk)
        # 取Top-1
        name_list=[]
        for i in range(topk):
            idx = int(indices[0][i])
            score = float(distances[0][i])
            matched_path = str(img_paths[idx])
            base = os.path.basename(matched_path)
            name, _ = os.path.splitext(base)
            name_list.append(name)
        return jsonify({"ok": True, "name": name_list, "score": score}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code or 500
    return jsonify({"ok": False, "error": str(e)}), code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=True)
