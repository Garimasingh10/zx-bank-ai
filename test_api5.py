import requests

resp = requests.get("https://openrouter.ai/api/v1/models")
models = resp.json()
if "data" in models:
    for m in models["data"]:
        if ("free" in m["id"] or m["pricing"]["prompt"] == "0") and "llama" in m["id"].lower():
            print(m["id"])
