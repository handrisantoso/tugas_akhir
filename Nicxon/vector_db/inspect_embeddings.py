#!/usr/bin/env python3
"""Quick inspection of new embedding file structures."""
import json

# Text embeddings
with open('embedding/text_embeddings_openrouter.json', 'r', encoding='utf-8') as f:
    text_embs = json.load(f)

print('=== TEXT EMBEDDINGS ===')
print('Count:', len(text_embs))
if text_embs:
    s = text_embs[0]
    print('Keys:', list(s.keys()))
    print('book_id:', s.get('book_id'))
    print('chunk_type:', s.get('chunk_type'))
    print('text (first 200):', s.get('text', '')[:200])
    print('embedding dim:', len(s.get('embedding', [])))
    meta = s.get('metadata', {})
    print('metadata keys:', list(meta.keys()))
    print('metadata sample:', {k: v for k, v in meta.items() if k != 'embedding_model'})

print()

# Image embeddings
with open('embedding/image_embeddings_openrouter.json', 'r', encoding='utf-8') as f:
    img_embs = json.load(f)

print('=== IMAGE EMBEDDINGS ===')
print('Count:', len(img_embs))
if img_embs:
    s = img_embs[0]
    print('Keys:', list(s.keys()))
    print('book_id:', s.get('book_id'))
    print('chunk_type:', s.get('chunk_type'))
    print('image_path:', s.get('image_path'))
    print('embedding dim:', len(s.get('embedding', [])))
    meta = s.get('metadata', {})
    print('metadata keys:', list(meta.keys()))
    print('metadata sample:', {k: v for k, v in meta.items() if k != 'embedding_model'})

print()

# Check overlap
text_ids = {e.get('book_id') for e in text_embs}
img_ids = {e.get('book_id') for e in img_embs}
print('=== OVERLAP ===')
print('Text book_ids:', len(text_ids))
print('Image book_ids:', len(img_ids))
print('Shared book_ids:', len(text_ids & img_ids))
print('Text only:', len(text_ids - img_ids))
print('Image only:', len(img_ids - text_ids))
