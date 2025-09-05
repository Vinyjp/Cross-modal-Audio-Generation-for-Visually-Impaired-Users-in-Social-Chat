# -*- coding: utf-8 -*-
"""
测试 Image Search API
用本地图片编码为 base64，POST 给 /api/search_image，打印返回的 basename。
"""
import base64, requests

API_URL = "http://127.0.0.1:7001/api/search_image"
TEST_IMAGE = "data/test.jpg"  # 替换为你的测试图片

def encode_file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    b64 = encode_file_to_b64(TEST_IMAGE)
    resp = requests.post(API_URL, json={"image_base64": b64, "topk": 1}, timeout=30)
    print("Status:", resp.status_code)
    print("JSON:", resp.json())

if __name__ == "__main__":
    main()
