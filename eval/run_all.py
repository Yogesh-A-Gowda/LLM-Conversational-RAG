"""
Runs the full evaluation suite and prints a presentation-ready summary:
  1. Retrieval quality  (eval/run_retrieval_eval.py)
  2. Generation quality (eval/run_generation_eval.py) -- LLM-judge based
  3. Latency benchmark  (eval/run_latency_bench.py)   -- Groq vs Ollama

Run: .venv/Scripts/python.exe -m eval.run_all [--gen-limit N] [--latency-limit N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_retrieval_eval, run_generation_eval, run_latency_bench


def main(gen_limit=None, latency_limit=8):
    print("\n" + "#" * 70)
    print("# 1/3  RETRIEVAL EVALUATION")
    print("#" * 70)
    retrieval_summary = run_retrieval_eval.main()

    print("\n" + "#" * 70)
    print("# 2/3  GENERATION EVALUATION (LLM-judge)")
    print("#" * 70)
    generation_summary = run_generation_eval.main(limit=gen_limit)

    print("\n" + "#" * 70)
    print("# 3/3  LATENCY BENCHMARK (Groq vs Ollama)")
    print("#" * 70)
    latency_summary = run_latency_bench.main(limit=latency_limit)

    print("\n" + "=" * 70)
    print("PRESENTATION SUMMARY")
    print("=" * 70)
    print(f"Retrieval  -- Hit Rate@{retrieval_summary['k']}: {retrieval_summary['hit_rate_at_k']:.1%}"
          f" | MRR: {retrieval_summary['mrr']:.3f}"
          f" | Precision@{retrieval_summary['k']}: {retrieval_summary['precision_at_k']:.1%}"
          f" | Scope-gate accuracy: {retrieval_summary['scope_gate_accuracy']:.1%}")
    print(f"Generation -- Faithfulness: {generation_summary['mean_faithfulness']:.1%}"
          f" | Answer Relevancy: {generation_summary['mean_answer_relevancy']:.1%}"
          f" | Context Precision: {generation_summary['mean_context_precision']:.1%}"
          f" | Context Recall (keyword proxy): {generation_summary['mean_context_recall_keyword']:.1%}")
    print(f"Latency    -- Retrieval: {latency_summary['retrieval']['mean_ms']}ms mean"
          f" | Groq generation: {latency_summary['groq_generation']['mean_ms']}ms mean"
          + (f" | Ollama generation: {latency_summary['ollama_generation']['mean_ms']}ms mean"
             if latency_summary.get("ollama_generation") else ""))
    print("\nFull per-question detail saved under eval/results/*.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-limit", type=int, default=None, help="Limit questions for generation eval (default: all)")
    parser.add_argument("--latency-limit", type=int, default=8, help="Limit questions for latency bench (default: 8)")
    args = parser.parse_args()
    main(gen_limit=args.gen_limit, latency_limit=args.latency_limit)
