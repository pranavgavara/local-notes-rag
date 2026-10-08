"""
eval.py - score the RAG pipeline against questions with known answers.

Usage:
  python eval.py sample-notes llama3.2
  python eval.py sample-notes llama3.2 mistral        (compare several models)

Each question in questions.json is checked automatically:
  must_include      - every one of these words must appear in the answer
  must_not_include  - none of these may appear (catches made-up details)
  refusal_expected  - the answer must admit the notes don't contain it
"""

import json
import sys
import time
from pathlib import Path

import ask_my_notes as rag

REFUSAL_PHRASES = ["no answer", "not in the notes", "not contain", "does not",
                   "do not", "doesn't", "don't", "cannot", "can't", "no mention",
                   "not mentioned", "not found", "not provided", "not specified",
                   "no information", "unable"]


def passed(case, answer):
    a = answer.lower()
    if case.get("refusal_expected"):
        return any(p in a for p in REFUSAL_PHRASES)
    if any(w.lower() not in a for w in case.get("must_include", [])):
        return False
    if any(w.lower() in a for w in case.get("must_not_include", [])):
        return False
    return True


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python eval.py <notes-folder> <model> [model2 ...]")
    folder = Path(sys.argv[1])
    models = sys.argv[2:]
    cases = json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8"))
    chunks, vectors = rag.build_index(folder)

    summary, details = [], []
    for model in models:
        rag.CHAT_MODEL = model
        score, start = 0, time.time()
        print(f"\n=== {model} ===")
        for case in cases:
            ans = rag.answer(case["q"], rag.retrieve(case["q"], chunks, vectors))
            ok = passed(case, ans)
            score += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {case['q']}")
            if not ok:
                print(f"       got: {ans[:200]!r}")
            details.append({"model": model, "question": case["q"], "pass": ok, "answer": ans})
        secs = time.time() - start
        summary.append((model, score, len(cases), secs))

    print("\n| Model | Score | Total time |\n|---|---|---|")
    for model, score, n, secs in summary:
        print(f"| {model} | {score}/{n} | {secs:.0f}s |")

    Path("eval_results.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("\nFull answers saved to eval_results.json")


if __name__ == "__main__":
    main()
