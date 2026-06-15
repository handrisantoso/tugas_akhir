#!/usr/bin/env python3
"""
Diagnostic script to find the exact multimodal embedding payload structure for OpenRouter.
"""

import json
import base64
import requests
from PIL import Image
import io

def create_tiny_image():
    # Create a tiny 8x8 red image
    img = Image.new('RGB', (8, 8), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64_data

def test_format(api_key, name, payload):
    print(f"\n--- Testing Format: {name} ---")
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Status Code: {resp.status_code}")
        try:
            body = resp.json()
            if resp.status_code == 200:
                print("✅ SUCCESS!")
                data = body.get("data", [])
                if data and len(data) > 0:
                    embedding = data[0].get("embedding", [])
                    print(f"   Embedding dimension: {len(embedding)}")
                else:
                    print(f"   Response body: {body}")
            else:
                print("❌ FAILED")
                print(f"   Response body: {json.dumps(body, indent=2)}")
        except Exception:
            print(f"❌ Could not parse JSON. Raw response: {resp.text[:500]}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def main():
    print("🚀 OpenRouter Multimodal Embedding Tester")
    print("========================================")
    api_key = input("🔑 Enter OpenRouter API key: ").strip()
    if not api_key:
        print("❌ API key required")
        return

    b64_raw = create_tiny_image()
    b64_data_url = f"data:image/png;base64,{b64_raw}"

    # Format 1: OpenAI Vision-style input array
    payload1 = {
        "model": "google/gemini-embedding-2-preview",
        "input": [
            {
                "type": "image_url",
                "image_url": {
                    "url": b64_data_url
                }
            }
        ]
    }

    # Format 2: Direct Base64 Data URL string as input
    payload2 = {
        "model": "google/gemini-embedding-2-preview",
        "input": b64_data_url
    }

    # Format 3: Voyage AI style dictionary
    payload3 = {
        "model": "google/gemini-embedding-2-preview",
        "input": {
            "image_url": b64_data_url
        }
    }

    # Format 4: Cohere style list
    payload4 = {
        "model": "google/gemini-embedding-2-preview",
        "input": [
            {
                "image": b64_data_url
            }
        ]
    }

    # Format 5: Google Gemini native content.parts structure (but via OpenRouter embeddings)
    payload5 = {
        "model": "google/gemini-embedding-2-preview",
        "content": {
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_raw
                    }
                }
            ]
        }
    }

    # Format 6: Simple list containing the base64 data url as a single string element
    payload6 = {
        "model": "google/gemini-embedding-2-preview",
        "input": [b64_data_url]
    }

    test_format(api_key, "1. OpenAI Vision-style [{{type: image_url, image_url: {{url: data:image/png;base64,...}}}}]", payload1)
    test_format(api_key, "2. Direct Base64 Data URL string as input", payload2)
    test_format(api_key, "3. Voyage-style {{image_url: data:image/png;base64,...}}", payload3)
    test_format(api_key, "4. Cohere-style [{{image: data:image/png;base64,...}}]", payload4)
    test_format(api_key, "5. Google Gemini native content.parts", payload5)
    test_format(api_key, "6. Simple list [data:image/png;base64,...]", payload6)

if __name__ == "__main__":
    main()
