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

import argparse
import json
import re
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
SEARCH_MODE = "vector"  # "vector" = meaning only, "hybrid" = meaning + keywords
NUM_CTX = 4096        # model context window in tokens; the prompt must fit or Ollama silently cuts it
TEMPERATURE = 0.0     # 0 = deterministic (same question, same answer): needed for fair comparisons
PROMPT_VERSION = "v1"
SYSTEM_PROMPTS = {
    "v1": "Answer using ONLY the notes provided. Do not add examples, details or advice that "
          "are not written in the notes. If the notes don't contain the answer, say so. "
          "Mention the source file name.",
    "v2": "Answer using ONLY the notes provided. Do not add examples, details or advice that "
          "are not written in the notes. Be specific: whenever the notes give a number, "
          "percentage, default value, name or field, state it exactly in your answer. "
          "Answer the question that was asked, not a nearby one. If the notes don't "
          "contain the answer, say so. Mention the source file name.",
    "v3": "Answer using ONLY the notes provided. Do not add details that are not written in the "
          "notes. Reply in exactly two parts. Part 1, a line starting 'Evidence:' that quotes the "
          "exact sentence(s) from the notes that bear on the question. Part 2, a line starting "
          "'Answer:' that uses those quotes to answer the question directly, stating every number, "
          "default value or name exactly. If a quoted sentence gives a value for a named item "
          "(for example a timer), use that value in the answer. If the notes don't contain the "
          "answer, write 'Answer: the notes do not contain this.' Mention the source file name.",
}


def clean_text(text: str) -> str:
    """Remove padding and page furniture so each chunk is mostly real content."""
    # drop RFC-style page footers/headers such as "[Page 4]" or "RFC 1918   Title   February 1996"
    lines = [ln for ln in text.splitlines()
             if not re.search(r"\[Page \d+\]", ln)
             and not re.match(r"^\s*RFC \d+\s.*\b(19|20)\d\d\s*$", ln)]
    text = "\n".join(lines)
    text = re.sub(r"[ \t\f\r]+", " ", text)      # runs of spaces/tabs -> one space
    text = re.sub(r"\n\s*\n+", "\n\n", text)     # many blank lines -> one
    return text.strip()


def chunk_text(text: str):
    text = clean_text(text)
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
    cache = folder / f".notes_index_{CHUNK_SIZE}_{CHUNK_OVERLAP}.json"   # one cache per chunk setting
    files = sorted(p for p in folder.rglob("*") if p.suffix in {".md", ".txt"})
    stamp = {str(p): p.stat().st_mtime for p in files}
    # also rebuild if chunking or the embedding model changes (bump "v2" if clean_text changes)
    stamp["_config"] = [CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, "v2"]

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


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


_BM25_CACHE = {}


def keyword_scores(question, chunks):
    """BM25: classic keyword ranking. Rewards chunks that contain the question's rarer words."""
    key = id(chunks)
    if key not in _BM25_CACHE:
        docs = [_tok(c["text"]) for c in chunks]
        df = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        _BM25_CACHE[key] = (docs, df, sum(len(d) for d in docs) / len(docs))
    docs, df, avgdl = _BM25_CACHE[key]
    n, k1, b = len(docs), 1.5, 0.75
    scores = np.zeros(n, dtype=np.float32)
    for w in set(_tok(question)):
        if w not in df:
            continue
        if n >= 20 and df[w] > 0.5 * n:   # skip words in most chunks ("the", "bgp"): no signal
            continue
        idf = np.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(w)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
    return scores


def retrieve(question, chunks, vectors):
    q = embed([question])[0]
    sims = (vectors @ q) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-9)
    if SEARCH_MODE == "hybrid":
        kw = keyword_scores(question, chunks)
        rank_v = np.empty(len(sims), dtype=int)
        rank_v[np.argsort(-sims)] = np.arange(len(sims))
        rank_k = np.empty(len(kw), dtype=int)
        rank_k[np.argsort(-kw)] = np.arange(len(kw))
        # reciprocal rank fusion; chunks with no keyword match get no keyword credit
        fused = 1 / (60 + rank_v) + np.where(kw > 0, 1 / (60 + rank_k), 0.0)
        best = np.argsort(-fused)[:TOP_K]
        return [dict(chunks[i], score=float(fused[i]), vec=float(sims[i]), kw=float(kw[i]))
                for i in best]
    best = np.argsort(sims)[::-1][:TOP_K]
    return [dict(chunks[i], score=float(sims[i])) for i in best]


def answer(question, context_chunks):
    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[PROMPT_VERSION]},
        {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"},
    ]
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model": CHAT_MODEL, "messages": messages, "stream": False,
                            "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "seed": 42}},
                      timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    global TOP_K, CHAT_MODEL, SEARCH_MODE, CHUNK_SIZE, CHUNK_OVERLAP, PROMPT_VERSION, NUM_CTX, TEMPERATURE
    parser =argparse.ArgumentParser(description="Ask questions about your own notes, locally.")
    parser.add_argument("folder", help="folder containing .md / .txt files")
    parser.add_argument("--k", type=int, default=TOP_K, help="chunks to retrieve (default %(default)s)")
    parser.add_argument("--model", default=CHAT_MODEL, help="Ollama chat model (default %(default)s)")
    parser.add_argument("--show", action="store_true", help="print the retrieved chunks and scores")
    parser.add_argument("--search", choices=["vector", "hybrid"], default=SEARCH_MODE,
                        help="vector = meaning only; hybrid = meaning + keywords (default %(default)s)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="characters per chunk (default %(default)s)")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="characters shared between chunks (default %(default)s)")
    parser.add_argument("--prompt", choices=sorted(SYSTEM_PROMPTS), default=PROMPT_VERSION,
                        help="system prompt version (default %(default)s)")
    parser.add_argument("--ctx", type=int, default=NUM_CTX, help="context window in tokens (default %(default)s)")
    parser.add_argument("--temp", type=float, default=TEMPERATURE, help="sampling temperature (default %(default)s)")
    args = parser.parse_args()
    TEMPERATURE = args.temp
    NUM_CTX = args.ctx
    PROMPT_VERSION = args.prompt
    TOP_K, CHAT_MODEL, SEARCH_MODE = args.k, args.model, args.search
    CHUNK_SIZE, CHUNK_OVERLAP = args.chunk_size, args.overlap

    chunks, vectors = build_index(Path(args.folder))
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        found = retrieve(q, chunks, vectors)
        if args.show:
            for c in found:
                if "vec" in c:
                    print(f"\n--- {c['source']}  (vector {c['vec']:.3f} | keyword {c['kw']:.1f}) ---")
                else:
                    print(f"\n--- {c['source']}  (similarity {c['score']:.3f}) ---")
                print(c["text"][:300].replace("\n", " ") + " ...")
        print("\n" + answer(q, found))


if __name__ == "__main__":
    main()
