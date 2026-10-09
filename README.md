# local-notes-rag

Ask questions about your own notes using an LLM that runs **entirely on your laptop**.
No API keys, no cloud, no per-token cost, and your notes never leave the machine.

Built as a learning project on a 2019 Dell XPS 15 (GTX 1650, 4 GB VRAM, 32 GB RAM) using
[Ollama](https://ollama.com). Beyond the pipeline itself, the repo includes a small evaluation
harness, because the most useful thing I learned is that a RAG system needs to be *measured*,
not just demoed.

## How it works

```
 your notes (.md / .txt)
        |
        v
 clean + split into overlapping chunks
        |
        v
 embedding model (nomic-embed-text) turns each chunk into a vector of numbers
        |   (cached to disk, so later runs start instantly)
        v
 you ask a question --> retrieve the best chunks
        |      vector search (meaning)  or  hybrid (meaning + BM25 keywords)
        v
 chat model (llama3.2 / mistral) answers using ONLY those chunks
```

This pattern is called RAG (retrieval-augmented generation): look up the relevant text first,
then let the model answer from it, instead of relying on what the model remembers.

## Setup

1. Install [Ollama](https://ollama.com), then pull the models:
   ```
   ollama pull llama3.2
   ollama pull mistral            # optional, for comparison
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

The first run embeds your files and caches the index in that folder. The index rebuilds
automatically when a file or a chunking setting changes.

## Options

| Flag | Meaning | Default |
|---|---|---|
| `--model` | Ollama chat model | llama3.2 |
| `--k` | chunks given to the model | 4 |
| `--search` | `vector` (meaning only) or `hybrid` (meaning + keywords) | vector |
| `--chunk-size`, `--overlap` | characters per chunk / shared between chunks | 800 / 150 |
| `--prompt` | system prompt version (`v1`, `v2`, `v3`) | v1 |
| `--ctx` | context window in tokens | 4096 |
| `--temp` | sampling temperature (0 = repeatable) | 0 |
| `--show` | print the retrieved chunks | off |

## Evaluating it

`eval.py` runs a file of questions with known answers and scores each reply automatically:

```
python eval.py sample-notes llama3.2 mistral
python eval.py "path\to\rfcs" llama3.2 --questions rfc_questions.json --search hybrid --k 8 --chunk-size 1500 --overlap 300
```

Each question can require words (`must_include`, `must_include_any`), forbid made-up ones
(`must_not_include`), or expect the model to admit the notes don't contain the answer
(`refusal_expected`). The harness also records whether *retrieval* found the needed text at all,
which separates "the search failed" from "the model failed". Runs use temperature 0 and a fixed
seed so comparisons are repeatable.

`rfc_questions.json` tests 11 questions against three public RFCs (1918, 2131, 4271), including
two questions whose answers are not in the text.

## Results (RFC test, 11 questions)

Hardware: Dell XPS 15 7590, GTX 1650 4 GB, 32 GB RAM. Hybrid search, k=8, 1500/300 chunks, temp 0.
Single runs, so one question is roughly 9 points of score.

| Model | Prompt | Score | Retrieval found the text | Total time |
|---|---|---|---|---|
| llama3.2 (3B) | v1 | 9/11 | 11/11 | 187s |
| llama3.2 (3B) | v3 (quote evidence, then answer) | 9/11 | 11/11 | 201s |
| mistral (7B) | v1 | 10/11 | 11/11 | 546s |

Raw speed: llama3.2 ~31 tokens/s (fits in 4 GB VRAM), mistral ~7 tokens/s (spills to CPU).

## What I learned

- **Retrieval and generation fail separately.** Early on, the right text often wasn't being
  retrieved at all. Measuring "did the search find it?" apart from "did the model answer
  correctly?" showed which half to fix.
- **Small changes move things in surprising ways.** Stripping RFC page headers and footers shifted
  chunk boundaries and broke a question that had passed. Adding keyword (BM25) search with
  rank fusion rescued it, but only after ignoring words that appear in most chunks.
- **Run-to-run noise hides real effects.** With default sampling, identical runs disagreed. Fixing
  temperature and seed made prompt and setting comparisons meaningful. Change one variable at a time.
- **Graders can be wrong too.** Plain substring matching said "half" was in "behalf" and treated
  "3" and "three" as different. I fixed the checker rather than blaming the model.
- **Multi-step questions are the weak spot.** Both models answer "what is the default for timer T2?"
  correctly, but fail "when does a client start rebinding?", where the answer needs linking
  "rebinding" to T2 and then T2 to 0.875 of the lease. The text was in the context every time.
  Mistral (larger) was one question better overall but still failed this one, and the
  "quote the evidence first" prompt fixed the renewing question but made the model invent
  an answer for a question that isn't in the notes (it traded one failure for another).
- **GPU memory decides speed.** A 3B model that fits in 4 GB of VRAM is about 4x faster than a
  7B model that spills onto the CPU, for roughly one more correct answer in eleven.

## Ideas to extend

- Rewrite the question into a keyword-style search query before retrieving (aimed at the multi-step failures)
- Try a larger model on the multi-step questions
- Add a simple web UI and run it as an always-on service on the home network
- Let the model use tools (read-only web search)
- Support PDFs and Word documents

## Files

| File | Purpose |
|---|---|
| `ask_my_notes.py` | The RAG pipeline |
| `eval.py` | Scores the pipeline against question files |
| `questions.json` | Questions for `sample-notes` |
| `rfc_questions.json` | Questions for the RFC test |
| `sample-notes/` | Fictional example notes for testing |
| `requirements.txt` | Python dependencies |
