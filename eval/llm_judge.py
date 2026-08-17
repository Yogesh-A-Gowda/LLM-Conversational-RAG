"""
Custom LLM-as-judge scorer, standing in for the `ragas` library (which
can't be installed in this environment without a C++ build toolchain).

Reproduces ragas' three headline generation metrics:
  - faithfulness       : is every claim in the answer grounded in the retrieved context?
  - answer_relevancy   : does the answer actually address the question asked?
  - context_precision  : of the retrieved chunks, how many are actually relevant?

Deliberately uses a DIFFERENT model as judge than the one that generated the
answer (see JUDGE_MODEL vs the generation model in rag_pipeline.generate_groq),
to reduce a model's tendency to rate its own output favorably.
"""
import json
import re
import time

import groq

JUDGE_MODEL = "openai/gpt-oss-20b"

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)(ms|s)", re.IGNORECASE)


def _call_judge(client, system_prompt: str, user_prompt: str, model: str = JUDGE_MODEL,
                 max_tokens: int = 2048, retries: int = 4) -> dict:
    last_error = None
    strict_json_mode = True
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + ("" if strict_json_mode else
                        "\n\nRespond with ONLY the JSON object, no other text.")},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if strict_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                raise
        except groq.RateLimitError as e:
            last_error = e
            match = _RETRY_AFTER_RE.search(str(e))
            if match:
                value, unit = float(match.group(1)), match.group(2).lower()
                wait_s = value / 1000 if unit == "ms" else value
            else:
                wait_s = 5.0
            time.sleep(wait_s + 0.5)
        except groq.BadRequestError as e:
            last_error = e
            if "json_validate_failed" in str(e):
                strict_json_mode = False
            time.sleep(1.5 * (attempt + 1))
        except json.JSONDecodeError as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Judge call failed after {retries} attempts: {last_error}")


def judge_faithfulness(client, answer: str, context: str) -> dict:
    system = (
        "You are a strict fact-checking grader. You will be given an AI-generated ANSWER "
        "and the source CONTEXT it was supposed to be based on. Break the answer into atomic "
        "factual claims, then decide for each claim whether it is directly supported by the "
        "context. Respond ONLY with JSON: "
        '{"total_claims": <int>, "supported_claims": <int>, "unsupported_claims": [<string>, ...]}'
    )
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    result = _call_judge(client, system, user)
    total = max(result.get("total_claims", 0), 1)
    result["score"] = result.get("supported_claims", 0) / total
    return result


def judge_answer_relevancy(client, question: str, answer: str) -> dict:
    system = (
        "You are grading how relevant an AI answer is to the question asked, on a 0.0-1.0 scale. "
        "1.0 = directly and completely addresses the question. 0.0 = does not address it at all "
        "(e.g. off-topic, or a refusal with no useful content). "
        'Respond ONLY with JSON: {"score": <float 0-1>, "reason": <short string>}'
    )
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return _call_judge(client, system, user)


def judge_context_precision(client, question: str, chunks: list) -> dict:
    n = len(chunks)
    schema = ", ".join(f'"{i + 1}": <0 or 1>' for i in range(n))
    system = (
        "You are grading retrieval quality. You will be given a QUESTION and exactly "
        f"{n} retrieved CHUNKS, numbered 1 to {n}. For EVERY chunk, decide if it is relevant (1) "
        f"or irrelevant (0) to answering the question. You must return a verdict for all {n} chunks "
        "-- no more, no fewer, keyed by chunk number as a string. "
        f'Respond ONLY with this JSON shape: {{"relevance": {{{schema}}}}}'
    )
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    user = f"QUESTION:\n{question}\n\nCHUNKS ({n} total):\n{numbered}"
    result = _call_judge(client, system, user)

    rel_map = result.get("relevance", {})
    if not isinstance(rel_map, dict):
        rel_map = {}
    verdicts = []
    for i in range(1, n + 1):
        v = rel_map.get(str(i), rel_map.get(i, 0))
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = 0
        verdicts.append(1 if v >= 1 else 0)

    result["relevance_by_chunk"] = verdicts
    result["score"] = (sum(verdicts) / n) if n else 0.0
    return result
