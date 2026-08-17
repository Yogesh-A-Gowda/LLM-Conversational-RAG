"""
Generation-quality evaluation using Ollama as the answer generator.

Same retrieval + prompt logic as run_generation_eval.py; only the generation
step is swapped from Groq to a local Ollama model.  The judge model stays on
Groq (openai/gpt-oss-20b) because Ollama has no equivalent judge available.

Run:
    .venv\\Scripts\\python.exe -m eval.run_generation_eval_ollama [--model llama3.1:8b] [--limit N]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics, llm_judge
from eval.golden_set import IN_SCOPE_QUESTIONS
from eval.rag_pipeline import (
    retrieve_with_scores, build_context, build_prompt,
    generate_ollama, get_groq_client, ollama_available, TOP_K,
)

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def main(model: str = DEFAULT_OLLAMA_MODEL, limit: int = None):
    if not ollama_available():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    client = get_groq_client()  # judge still uses Groq
    questions = IN_SCOPE_QUESTIONS[:limit] if limit else IN_SCOPE_QUESTIONS
    per_question = []

    for item in questions:
        print(f"[{item['id']}] Generating with Ollama ({model})...")
        docs_with_scores, _ = retrieve_with_scores(item["question"], k=TOP_K)
        docs = [d for d, _ in docs_with_scores]
        context = build_context(docs)
        prompt = build_prompt(item["question"], context, "(No prior conversation yet.)")

        try:
            answer, gen_ms = generate_ollama(prompt, model=model)
        except Exception as e:
            print(f"  Ollama generation failed: {e}")
            per_question.append({
                "id": item["id"], "question": item["question"], "answer": None,
                "generation_ms": None, "ollama_error": str(e),
                "faithfulness": None, "answer_relevancy": None,
                "context_recall_keyword": None,
            })
            continue

        try:
            faithfulness = llm_judge.judge_faithfulness(client, answer, context)
            relevancy = llm_judge.judge_answer_relevancy(client, item["question"], answer)
        except Exception as e:
            print(f"  Judge failed: {e}")
            per_question.append({
                "id": item["id"], "question": item["question"], "answer": answer,
                "generation_ms": round(gen_ms, 1), "judge_error": str(e),
                "faithfulness": None, "answer_relevancy": None,
                "context_recall_keyword": metrics.keyword_recall(context, item["expected_keywords"])[0],
            })
            continue

        recall_score, _ = metrics.keyword_recall(context, item["expected_keywords"])

        row = {
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "generation_ms": round(gen_ms, 1),
            "faithfulness": faithfulness["score"],
            "faithfulness_unsupported_claims": faithfulness.get("unsupported_claims", []),
            "answer_relevancy": relevancy.get("score", 0.0),
            "context_recall_keyword": recall_score,
        }
        per_question.append(row)
        print(f"  faith={row['faithfulness']:.2f} relevancy={row['answer_relevancy']:.2f} "
              f"ctx_recall={row['context_recall_keyword']:.2f} :: {item['question'][:55]}")

    summary = {
        "n_questions": len(per_question),
        "generation_model": model,
        "generation_backend": "ollama",
        "judge_model": llm_judge.JUDGE_MODEL,
        "mean_faithfulness": metrics.mean([r["faithfulness"] for r in per_question]),
        "mean_answer_relevancy": metrics.mean([r["answer_relevancy"] for r in per_question]),
        "mean_context_recall_keyword": metrics.mean([r["context_recall_keyword"] for r in per_question]),
        "mean_generation_ms": metrics.mean([r["generation_ms"] for r in per_question]),
    }

    print("-" * 70)
    print(f"Mean faithfulness       : {summary['mean_faithfulness']:.1%}")
    print(f"Mean answer relevancy   : {summary['mean_answer_relevancy']:.1%}")
    print(f"Mean context recall*    : {summary['mean_context_recall_keyword']:.1%}  (*keyword-coverage proxy)")
    print(f"Mean generation latency : {summary['mean_generation_ms']:.0f} ms  (model: {model})")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_metrics_ollama.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)
    print(f"\nSaved -> eval/results/generation_metrics_ollama.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name (default: llama3.1:8b)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions for a quick run")
    args = parser.parse_args()
    main(model=args.model, limit=args.limit)
