# -*- coding: utf-8 -*-
"""
测试 Audio Search API
向 /api/search_audio 发送 text（通常是图片的 basename），打印返回的音频路径。
"""
import requests

API_URL = "http://127.0.0.1:7002/api/search_audio"
TEST_TEXT = "affection1 character"  # 替换为上一步得到的 name

def main():
    resp = requests.post(API_URL, json={"text": TEST_TEXT, "topk": 1}, timeout=30)
    print("Status:", resp.status_code)
    print("JSON:", resp.json())

if __name__ == "__main__":
    main()
