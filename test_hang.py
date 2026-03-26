import requests
import json

payload = {"session_id": "user123", "query": "I am frustrated and want to talk to a human. My name is John Doe and my number is 555-0199."}
try:
    resp = requests.post("http://localhost:8000/chat", json=payload, timeout=5)
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print("FAILED:", e)
