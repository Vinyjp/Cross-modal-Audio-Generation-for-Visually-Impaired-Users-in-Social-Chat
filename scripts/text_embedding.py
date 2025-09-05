import requests

def get_embedding(text, model="Qwen/Qwen3-Embedding-8B", dimensions=768):
    url = "https://api.siliconflow.cn/v1/embeddings"
    model = "Qwen/Qwen3-Embedding-8B" # 可选的Qwen/Qwen3-Embedding-4B Qwen/Qwen3-Embedding-0.6B
    input_text = text

    payload = {
        "model": model,
        "input": input_text,
        "dimensions": dimensions
    }
    headers = {
        "Authorization": "Bearer sk-ivjnxwgjseyqhnreysclrlvxkjvmgrctwhaopvdoqcnzctok",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    vector = response.json()['data'][0]['embedding']
    return vector

vector = get_embedding("happy", "Qwen/Qwen3-Embedding-8B", 768)
print(vector)