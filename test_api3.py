import requests, os
from dotenv import load_dotenv

load_dotenv("d:\\yellow.ai\\.env")
key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {key}",
    "HTTP-Referer": "http://localhost:8000",
    "Content-Type": "application/json"
}
models = ["mistralai/mistral-7b-instruct:free", "meta-llama/llama-3-8b-instruct:free", "google/gemini-2.0-flash-exp:free", "google/gemini-2.0-pro-exp-02-05:free"]

for m in models:
    payload = {
        "model": m,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    print(m, resp.status_code)
