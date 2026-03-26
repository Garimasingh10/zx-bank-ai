import requests, os
from dotenv import load_dotenv

load_dotenv("d:\\yellow.ai\\.env")
key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {key}",
    "HTTP-Referer": "http://localhost:8000",
    "Content-Type": "application/json"
}
payload = {
    "model": "meta-llama/llama-3.2-3b-instruct:free",
    "messages": [
        {"role": "user", "content": "Say hello"}
    ],
    "max_tokens": 10
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
print("Status Code:", response.status_code)
if response.status_code != 200:
    print("Error:", response.text)
else:
    print("Success:", response.json()["choices"][0]["message"]["content"])
