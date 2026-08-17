"""
Latency benchmark: Groq (cloud) vs Ollama (local) generation, on identical
retrieval + prompt for each question, using the same code path as the app.

Reports mean / median / p95 for:
  - retrieval time (Chroma similarity search -- same for both backends)
  - generation time per backend/model

Run: .venv/Scripts/python.exe -m eval.run_latency_bench [--limit N] [--ollama-model NAME]
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.golden_set import IN_SCOPE_QUESTIONS
from eval.rag_pipeline import (
    retrieve_with_scores, build_context, build_prompt,
    generate_groq, generate_ollama, ollama_available, TOP_K,
)

GROQ_MODEL = "llama-3.3-70b-versatile"


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(round(pct / 100 * (len(s) - 1))), len(s) - 1)
    return s[idx]


def stats_block(values):
    return {
        "mean_ms": round(statistics.mean(values), 1) if values else None,
        "median_ms": round(statistics.median(values), 1) if values else None,
        "p95_ms": round(percentile(values, 95), 1) if values else None,
        "n": len(values),
    }


def main(limit=None, ollama_model="llama3.1:8b"):
    questions = IN_SCOPE_QUESTIONS[:limit] if limit else IN_SCOPE_QUESTIONS
    has_ollama = ollama_available()
    if not has_ollama:
        print("Ollama not reachable at localhost:11434 -- skipping local-model comparison, Groq only.")

    retrieval_times, groq_times, ollama_times = [], [], []
    rows = []

    for item in questions:
        docs_with_scores, retrieval_ms = retrieve_with_scores(item["question"], k=TOP_K)
        docs = [d for d, _ in docs_with_scores]
        context = build_context(docs)
        prompt = build_prompt(item["question"], context, "(No prior conversation yet.)")
        retrieval_times.append(retrieval_ms)

        _, groq_ms = generate_groq(prompt, model=GROQ_MODEL)
        groq_times.append(groq_ms)

        row = {"id": item["id"], "retrieval_ms": round(retrieval_ms, 1), "groq_ms": round(groq_ms, 1)}

        if has_ollama:
            try:
                _, ollama_ms = generate_ollama(prompt, model=ollama_model)
                ollama_times.append(ollama_ms)
                row["ollama_ms"] = round(ollama_ms, 1)
            except Exception as e:
                row["ollama_ms"] = None
                row["ollama_error"] = str(e)

        rows.append(row)
        print(f"[{item['id']}] retrieval={row['retrieval_ms']:.0f}ms  groq={row['groq_ms']:.0f}ms  "
              f"ollama={row.get('ollama_ms', 'n/a')}")

    summary = {
        "n_questions": len(rows),
        "groq_model": GROQ_MODEL,
        "ollama_model": ollama_model if has_ollama else None,
        "retrieval": stats_block(retrieval_times),
        "groq_generation": stats_block(groq_times),
        "ollama_generation": stats_block(ollama_times) if has_ollama else None,
    }

    print("-" * 70)
    print(f"Retrieval        : mean={summary['retrieval']['mean_ms']}ms  median={summary['retrieval']['median_ms']}ms  p95={summary['retrieval']['p95_ms']}ms")
    print(f"Groq generation  ({GROQ_MODEL}): mean={summary['groq_generation']['mean_ms']}ms  median={summary['groq_generation']['median_ms']}ms  p95={summary['groq_generation']['p95_ms']}ms")
    if has_ollama and summary["ollama_generation"]["n"] > 0:
        og = summary["ollama_generation"]
        print(f"Ollama generation ({ollama_model}): mean={og['mean_ms']}ms  median={og['median_ms']}ms  p95={og['p95_ms']}ms")
        speedup = og["mean_ms"] / summary["groq_generation"]["mean_ms"]
        print(f"Groq is ~{speedup:.1f}x faster than local Ollama ({ollama_model}) for generation on this machine.")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": rows}, f, indent=2)
    print(f"\nSaved -> eval/results/latency_benchmark.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8, help="Number of questions to benchmark (default 8)")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    args = parser.parse_args()
    main(limit=args.limit, ollama_model=args.ollama_model)
