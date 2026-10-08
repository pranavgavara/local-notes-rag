# local-notes-rag

Ask questions about your own notes using an LLM that runs **entirely on your laptop**.
No API keys, no cloud, no per-token cost, and your notes never leave the machine.

Built as a learning project on a 2019 Dell XPS 15 (GTX 1650, 4 GB VRAM, 32 GB RAM) using
[Ollama](https://ollama.com).

## How it works

```
 your notes (.md / .txt)
        |
        v
 split into overlapping chunks  (~800 characters each)
        |
        v
 embedding model (nomic-embed-text) turns each chunk into a vector of numbers
        |   (cached to disk, so later runs start instantly)
        v
 you ask a question --> the question is embedded the same way
        |
        v
 cosine similarity picks the 4 most similar chunks
        |
        v
 chat model (llama3.2) answers using ONLY those chunks
```

This pattern is called RAG (retrieval-augmented generation): look up the relevant text first,
then let the model answer from it, instead of relying on what the model remembers.

## Setup

1. Install [Ollama](https://ollama.com), then pull two models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. Install the Python packages (Python 3.9+):
   ```
   pip install -r requirements.txt
   ```
3. Run it on a folder of `.md` / `.txt` files:
   ```
   python ask_my_notes.py sample-notes
   ```

The first run embeds your files and saves `.notes_index.json` in that folder. The index rebuilds
automatically when a file changes.

## Benchmarks

Hardware: Dell XPS 15 7590 (2019), NVIDIA GTX 1650 4 GB, 32 GB RAM, Windows.
Measured with `ollama run <model> --verbose`, one short prompt each ("explain what a subnet mask
is in two sentences"). Single runs, so treat them as rough numbers.

| Model | Size | Generation speed | Notes |
|---|---|---|---|
| llama3.2 | 3B | ~31 tokens/s | Fits entirely in the GPU's 4 GB VRAM |
| mistral | 7B | ~7 tokens/s | Too big for 4 GB VRAM, so it is split between GPU and CPU |

## Test results (sample-notes)

| Question | Result |
|---|---|
| What is the NAS called and what is its IP? | Correct (vault, 192.168.1.50) |
| What accuracy did my cricket classifier reach? | Correct (94%) |
| Why did my model fail on grass photos? | Correct, cited the source file |
| What is my favourite cricket team? (not in the notes) | Correctly said it could not find an answer |

## What I learned

- With the first version of the prompt, the 3B model added details that were not in the notes
  (it listed "random cropping or rotation" when the note only said "data augmentation").
  The prompt now tells it not to add anything beyond the notes. Small models follow
  instructions loosely, so always test with questions whose answers are NOT in the notes.
- GPU memory decides speed. A 3B model fits in 4 GB of VRAM and is about 4x faster than a 7B
  model that has to spill onto the CPU.

## Ideas to extend

- Compare chunk size, `TOP_K` and chat model (llama3.2 vs mistral) on answer quality
- Add a simple web UI
- Support PDFs and Word documents
- Show which chunks were retrieved alongside each answer
- Build a small test set of questions and score accuracy automatically

## Files

| File | Purpose |
|---|---|
| `ask_my_notes.py` | The whole pipeline (about 100 lines) |
| `sample-notes/` | Fictional example notes for testing |
| `requirements.txt` | Python dependencies |


# Evaluation
 | Model | Score | Total time |
|---|---|---|
| llama3.2 | 9/9 | 67s |
| mistral | 9/9 | 113s |