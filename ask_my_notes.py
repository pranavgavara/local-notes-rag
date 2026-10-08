"""
ask_my_notes.py - a minimal local RAG pipeline using Ollama.

What it does:
  1. Reads every .md / .txt file in a folder
  2. Splits them into overlapping chunks
  3. Embeds each chunk with a local embedding model (cached to disk)
  4. For each question: embeds the question, finds the most similar chunks,
     and passes them to a local chat model as context

Setup (one time, in PowerShell):
  ollama pull nomic-embed-text
  ollama pull llama3.2
  pip install requests numpy

Usage:
  python ask_my_notes.py "C:\\path\\to\\notes"
Then type questions. Ctrl+C to quit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # how many chunks to give the model


def chunk_text(text: str):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=np.float32)


def build_index(folder: Path):
    cache = folder / ".notes_index.json"
    files = sorted(p for p in folder.rglob("*") if p.suffix in {".md", ".txt"})
    stamp = {str(p): p.stat().st_mtime for p in files}

    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["stamp"] == stamp:
            print("Loaded cached index.")
            return saved["chunks"], np.array(saved["vectors"], dtype=np.float32)

    chunks = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_text(text):
            chunks.append({"source": p.name, "text": c})
    if not chunks:
        sys.exit("No .md or .txt files found in that folder.")

    print(f"Embedding {len(chunks)} chunks from {len(files)} files...")
    vectors = []
    for i in range(0, len(chunks), 32):  # batch to keep requests small
        vectors.append(embed([c["text"] for c in chunks[i:i + 32]]))
    vectors = np.vstack(vectors)

    cache.write_text(json.dumps({"stamp": stamp, "chunks": chunks,
                                 "vectors": vectors.tolist()}), encoding="utf-8")
    return chunks, vectors


def retrieve(question, chunks, vectors):
    q = embed([question])[0]
    sims = (vectors @ q) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-9)
    best = np.argsort(sims)[::-1][:TOP_K]
    return [chunks[i] for i in best]


def answer(question, context_chunks):
    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in context_chunks)
    messages = [
        {"role": "system",
         "content": "Answer using ONLY the notes provided. Do not add examples, "
                    "details or advice that are not written in the notes. If the "
                    "notes don't contain the answer, say so. Mention the source "
                    "file name."},
        {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"},
    ]
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model": CHAT_MODEL, "messages": messages, "stream": False},
                      timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python ask_my_notes.py "C:\\path\\to\\notes"')
    folder = Path(sys.argv[1])
    chunks, vectors = build_index(folder)
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            print("\n" + answer(q, retrieve(q, chunks, vectors)))


if __name__ == "__main__":
    main()
