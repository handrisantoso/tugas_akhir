import requests
import json

key = ""

models = [
    "google/gemini-embedding-001",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "qwen/qwen3-embedding-8b",
    "qwen/qwen3-embedding-0.6b",
]

for model in models:
    r = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "input": "hello world"}
    )
    resp = r.json()
    if "data" in resp and resp["data"]:
        dim = len(resp["data"][0]["embedding"])
        print(f"  {model:45s} -> WORKING  dim={dim}")
    else:
        err = resp.get("error", {}).get("message", "unknown")
        print(f"  {model:45s} -> FAILED   ({err})")
