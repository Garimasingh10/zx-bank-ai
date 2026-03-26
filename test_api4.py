import requests

resp = requests.get("https://openrouter.ai/api/v1/models")
models = resp.json()["data"]
for m in models:
    if "free" in m["id"] or m["pricing"]["prompt"] == "0":
        print(m["id"])
        break
