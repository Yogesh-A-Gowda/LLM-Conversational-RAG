"""
Retrieval-quality evaluation for the LectureBank RAG app.

Metrics (no LLM calls -- purely computed from what Chroma actually returns):
  - Hit Rate@5   : fraction of questions where a known-correct chunk appears in top-5
  - MRR          : mean reciprocal rank of the first correct chunk
  - Precision@5  : average fraction of the top-5 chunks that are correct
  - Scope-gate accuracy : does the 0.3 relevance-score cutoff (app.py MIN_SCOPE_RELEVANCE)
                          correctly accept in-scope questions and reject out-of-scope ones?

Run: .venv/Scripts/python.exe -m eval.run_retrieval_eval
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics
from eval.golden_set import IN_SCOPE_QUESTIONS, OUT_OF_SCOPE_QUESTIONS
from eval.rag_pipeline import retrieve_with_scores, detect_scope, TOP_K


def main():
    per_question = []

    for item in IN_SCOPE_QUESTIONS:
        docs_with_scores, retrieval_ms = retrieve_with_scores(item["question"], k=TOP_K)
        docs = [d for d, _ in docs_with_scores]
        result = metrics.evaluate_retrieval(docs, item["matchers"], k=TOP_K)
        in_scope = detect_scope(docs_with_scores)
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "type": "in_scope",
            "scope_gate_correct": in_scope is True,
            "retrieval_ms": round(retrieval_ms, 1),
            **result,
        })

    for item in OUT_OF_SCOPE_QUESTIONS:
        docs_with_scores, retrieval_ms = retrieve_with_scores(item["question"], k=TOP_K)
        in_scope = detect_scope(docs_with_scores)
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "type": "out_of_scope",
            "scope_gate_correct": in_scope is False,
            "retrieval_ms": round(retrieval_ms, 1),
            "hit_at_k": None,
            "reciprocal_rank": None,
            "precision_at_k": None,
        })

    in_scope_rows = [r for r in per_question if r["type"] == "in_scope"]
    summary = {
        "n_in_scope_questions": len(in_scope_rows),
        "n_out_of_scope_questions": len(per_question) - len(in_scope_rows),
        "hit_rate_at_k": metrics.mean([r["hit_at_k"] for r in in_scope_rows]),
        "mrr": metrics.mean([r["reciprocal_rank"] for r in in_scope_rows]),
        "precision_at_k": metrics.mean([r["precision_at_k"] for r in in_scope_rows]),
        "scope_gate_accuracy": metrics.mean([1.0 if r["scope_gate_correct"] else 0.0 for r in per_question]),
        "mean_retrieval_ms": metrics.mean([r["retrieval_ms"] for r in per_question]),
        "k": TOP_K,
    }

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    for r in per_question:
        if r["type"] == "in_scope":
            print(f"[{r['id']}] hit@{TOP_K}={r['hit_at_k']!s:5} rank={str(r['first_rank']):4} "
                  f"prec@{TOP_K}={r['precision_at_k']:.2f} scope_ok={r['scope_gate_correct']} :: {r['question'][:60]}")
        else:
            print(f"[{r['id']}] (out-of-scope) scope_ok={r['scope_gate_correct']} :: {r['question'][:60]}")

    print("-" * 70)
    print(f"Hit Rate@{TOP_K}         : {summary['hit_rate_at_k']:.1%}")
    print(f"MRR                     : {summary['mrr']:.3f}")
    print(f"Precision@{TOP_K}        : {summary['precision_at_k']:.1%}")
    print(f"Scope-gate accuracy     : {summary['scope_gate_accuracy']:.1%}  ({summary['n_in_scope_questions']} in-scope + {summary['n_out_of_scope_questions']} out-of-scope)")
    print(f"Mean retrieval latency  : {summary['mean_retrieval_ms']:.1f} ms")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retrieval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)
    print(f"\nSaved -> eval/results/retrieval_metrics.json")

    return summary


if __name__ == "__main__":
    main()
