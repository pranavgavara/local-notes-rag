"""
eval.py - score the RAG pipeline against questions with known answers.

Usage:
  python eval.py sample-notes llama3.2
  python eval.py sample-notes llama3.2 mistral                 (compare models)
  python eval.py "Real Notes" llama3.2 --questions rfc_questions.json
  python eval.py "Real Notes" llama3.2 --search vector         (compare search modes)
  python eval.py "Real Notes" llama3.2 --search hybrid --k 8

Each question in the questions file is checked automatically:
  must_include      - every one of these words must appear in the answer
  must_include_any  - at least one of these must appear (e.g. "0.5" or "50%")
  must_not_include  - none of these may appear (catches made-up details)
  refusal_expected  - the answer must admit the notes don't contain it
Also recorded: retrieved_ok - did the retrieved chunks contain the needed words at all?
  (separates "search failed to find it" from "model failed to use it")
"""

import argparse
import json
import re
import time
from pathlib import Path

import ask_my_notes as rag

# A refusal = "no answer", or a negative word followed (in the same sentence) by a word like
# present / mention / provide / contain / found / stated ... e.g. "This information is not present
# in the notes", "The notes do not provide", "No information ... is provided".
REFUSAL_RE = re.compile(
    r"no answer|\b(?:no|not|cannot|can't|unable|isn't|doesn't|don't)\b[^.\n]{0,60}?"
    r"(?:present|mention|provid|contain|found|stated|specif|available|information|notes|source)",
    re.I)


def has(term, text):
    """Whole-term match: 'half' must not match 'behalf', '0.5' must not match '10.0.0.5'."""
    start = r"(?<![\w.])" if term[0].isalnum() else ""
    end = r"(?!\d)" if term[-1].isdigit() else ""
    return re.search(start + re.escape(term.lower()) + end, text.lower()) is not None


def contains_needed(case, text):
    """True if `text` holds every must_include term and at least one must_include_any term."""
    if any(not has(w, text) for w in case.get("must_include", [])):
        return False
    any_of = case.get("must_include_any")
    if any_of and not any(has(w, text) for w in any_of):
        return False
    return True


def passed(case, answer):
    if case.get("refusal_expected"):
        return REFUSAL_RE.search(answer) is not None
    if not contains_needed(case, answer):
        return False
    if any(has(w, answer) for w in case.get("must_not_include", [])):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Score the RAG pipeline on a question file.")
    ap.add_argument("folder", help="notes folder")
    ap.add_argument("models", nargs="+", help="one or more Ollama chat models")
    ap.add_argument("--questions", default="questions.json", help="question file (default %(default)s)")
    ap.add_argument("--search", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--k", type=int, default=rag.TOP_K, help="chunks to retrieve")
    ap.add_argument("--prompt", choices=sorted(rag.SYSTEM_PROMPTS), default=rag.PROMPT_VERSION,
                    help="system prompt version (default %(default)s)")
    ap.add_argument("--ctx", type=int, default=rag.NUM_CTX, help="context window in tokens")
    ap.add_argument("--temp", type=float, default=rag.TEMPERATURE, help="sampling temperature (0 = repeatable)")
    ap.add_argument("--chunk-size", type=int, default=rag.CHUNK_SIZE, help="characters per chunk")
    ap.add_argument("--overlap", type=int, default=rag.CHUNK_OVERLAP, help="characters shared between chunks")
    args = ap.parse_args()

    rag.SEARCH_MODE, rag.TOP_K = args.search, args.k
    rag.CHUNK_SIZE, rag.CHUNK_OVERLAP = args.chunk_size, args.overlap
    rag.PROMPT_VERSION, rag.NUM_CTX, rag.TEMPERATURE = args.prompt, args.ctx, args.temp
    qfile = Path(args.questions)
    if not qfile.is_absolute() and not qfile.exists():
        qfile = Path(__file__).with_name(args.questions)
    cases = json.loads(qfile.read_text(encoding="utf-8"))
    chunks, vectors = rag.build_index(Path(args.folder))

    # retrieval is the same for every model, so do it once per question
    retrieved = {c["q"]: rag.retrieve(c["q"], chunks, vectors) for c in cases}
    found = {c["q"]: (c.get("refusal_expected") or
                      contains_needed(c, " ".join(x["text"] for x in retrieved[c["q"]])))
             for c in cases}

    summary, details = [], []
    for model in args.models:
        rag.CHAT_MODEL = model
        score, start = 0, time.time()
        print(f"\n=== {model}  (search={args.search}, k={args.k}, "
              f"chunk={args.chunk_size}/{args.overlap}, prompt={args.prompt}, ctx={args.ctx}, "
              f"temp={args.temp}) ===")
        for case in cases:
            ans = rag.answer(case["q"], retrieved[case["q"]])
            ok = passed(case, ans)
            score += ok
            note = "" if found[case["q"]] else "  [search did not retrieve the needed text]"
            print(f"[{'PASS' if ok else 'FAIL'}] {case['q']}{note}")
            if not ok:
                print(f"       got: {ans[:200]!r}")
            details.append({"model": model, "question": case["q"], "pass": ok,
                            "retrieved_ok": bool(found[case["q"]]), "answer": ans,
                            "retrieved": [{"source": x["source"], "score": round(x["score"], 4),
                                           "text": x["text"]} for x in retrieved[case["q"]]]})
        summary.append((model, score, len(cases), time.time() - start))

    retrieval_hits = sum(bool(v) for v in found.values())
    print(f"\nRetrieval found the needed text for {retrieval_hits}/{len(cases)} questions "
          f"(search={args.search}, k={args.k}, chunk={args.chunk_size}/{args.overlap}, prompt={args.prompt}, ctx={args.ctx}, temp={args.temp})")
    print("\n| Model | Score | Total time |\n|---|---|---|")
    for model, score, n, secs in summary:
        print(f"| {model} | {score}/{n} | {secs:.0f}s |")

    Path("eval_results.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("\nFull answers saved to eval_results.json")


if __name__ == "__main__":
    main()
