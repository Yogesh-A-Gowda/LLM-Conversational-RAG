"""
Generation-quality evaluation for the LectureBank RAG app.

Runs the exact retrieval + prompt-building logic from app.py, generates a
real answer with the same model app.py defaults to (llama-3.3-70b-versatile
on Groq), then scores it on:

  - faithfulness       (LLM judge)  : is every claim grounded in retrieved context?
  - answer_relevancy   (LLM judge)  : does the answer address the question?
  - context_precision  (LLM judge)  : are the retrieved chunks actually relevant?
  - context_recall     (deterministic) : fraction of expected topical keywords
                                          present in the retrieved context
                                          (no gold reference answers exist for
                                          this dataset, so this is a keyword-
                                          coverage proxy, not ragas' LLM-based
                                          context_recall)

The judge model (openai/gpt-oss-120b) is deliberately different from the
generation model to reduce self-grading bias.

Run: .venv/Scripts/python.exe -m eval.run_generation_eval [--limit N]
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
    generate_groq, get_groq_client, TOP_K,
)

GENERATION_MODEL = "llama-3.3-70b-versatile"


def main(limit=None):
    client = get_groq_client()
    questions = IN_SCOPE_QUESTIONS[:limit] if limit else IN_SCOPE_QUESTIONS
    per_question = []

    for item in questions:
        docs_with_scores, _ = retrieve_with_scores(item["question"], k=TOP_K)
        docs = [d for d, _ in docs_with_scores]
        context = build_context(docs)
        prompt = build_prompt(item["question"], context, "(No prior conversation yet.)")
        answer, gen_ms = generate_groq(prompt, model=GENERATION_MODEL)

        try:
            faithfulness = llm_judge.judge_faithfulness(client, answer, context)
            relevancy = llm_judge.judge_answer_relevancy(client, item["question"], answer)
            precision = llm_judge.judge_context_precision(client, item["question"], [d.page_content for d in docs])
        except Exception as e:
            print(f"[{item['id']}] JUDGE FAILED, skipping this question's LLM-judge scores: {e}")
            per_question.append({
                "id": item["id"], "question": item["question"], "answer": answer,
                "generation_ms": round(gen_ms, 1), "judge_error": str(e),
                "faithfulness": None, "answer_relevancy": None, "context_precision": None,
                "context_recall_keyword": metrics.keyword_recall(context, item["expected_keywords"])[0],
            })
            continue

        recall_score, recall_found = metrics.keyword_recall(context, item["expected_keywords"])

        row = {
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "generation_ms": round(gen_ms, 1),
            "faithfulness": faithfulness["score"],
            "faithfulness_unsupported_claims": faithfulness.get("unsupported_claims", []),
            "answer_relevancy": relevancy.get("score", 0.0),
            "context_precision": precision["score"],
            "context_recall_keyword": recall_score,
        }
        per_question.append(row)
        print(f"[{row['id']}] faith={row['faithfulness']:.2f} relevancy={row['answer_relevancy']:.2f} "
              f"ctx_prec={row['context_precision']:.2f} ctx_recall={row['context_recall_keyword']:.2f} "
              f":: {item['question'][:55]}")

    summary = {
        "n_questions": len(per_question),
        "generation_model": GENERATION_MODEL,
        "judge_model": llm_judge.JUDGE_MODEL,
        "mean_faithfulness": metrics.mean([r["faithfulness"] for r in per_question]),
        "mean_answer_relevancy": metrics.mean([r["answer_relevancy"] for r in per_question]),
        "mean_context_precision": metrics.mean([r["context_precision"] for r in per_question]),
        "mean_context_recall_keyword": metrics.mean([r["context_recall_keyword"] for r in per_question]),
        "mean_generation_ms": metrics.mean([r["generation_ms"] for r in per_question]),
    }

    print("-" * 70)
    print(f"Mean faithfulness       : {summary['mean_faithfulness']:.1%}")
    print(f"Mean answer relevancy   : {summary['mean_answer_relevancy']:.1%}")
    print(f"Mean context precision  : {summary['mean_context_precision']:.1%}")
    print(f"Mean context recall*    : {summary['mean_context_recall_keyword']:.1%}  (*keyword-coverage proxy)")
    print(f"Mean generation latency : {summary['mean_generation_ms']:.0f} ms  (model: {GENERATION_MODEL})")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "generation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)
    print(f"\nSaved -> eval/results/generation_metrics.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for a quick run)")
    args = parser.parse_args()
    main(limit=args.limit)
