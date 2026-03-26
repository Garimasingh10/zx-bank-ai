import requests, os
from dotenv import load_dotenv

load_dotenv("d:\\yellow.ai\\.env")
key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {key}",
    "HTTP-Referer": "http://localhost:8000",
    "Content-Type": "application/json"
}

# Exhaustive list of free models
models = [
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "openchat/openchat-7b:free",
    "qwen/qwen-2-72b-instruct:free",
    "cognitivecomputations/dolphin-mixtral-8x7b:free"
]

working_model = None
for m in models:
    payload = {"model": m, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"SUCCESS: {m}")
            working_model = m
            break
        else:
            print(f"FAILED {m}: {resp.status_code}")
    except Exception as e:
        print(f"ERROR {m}: {e}")

if not working_model:
    print("NO FREE MODELS WORKED")
