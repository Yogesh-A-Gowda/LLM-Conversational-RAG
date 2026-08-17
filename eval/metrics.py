"""Deterministic (non-LLM) retrieval metrics."""


def is_relevant(metadata: dict, matchers: list) -> bool:
    for m in matchers:
        if "id" in m and str(metadata.get("id", "")) == str(m["id"]):
            return True
        if "source_file" in m and metadata.get("source_file") == m["source_file"]:
            return True
        if "title_contains" in m:
            title = (metadata.get("title") or "").lower()
            if m["title_contains"].lower() in title:
                return True
    return False


def evaluate_retrieval(docs, matchers: list, k: int = 5) -> dict:
    """Hit rate / MRR / precision@k for one question's retrieved docs."""
    top_k = docs[:k]
    hits = [is_relevant(d.metadata, matchers) for d in top_k]
    first_rank = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {
        "hit_at_k": any(hits),
        "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
        "first_rank": first_rank,
        "precision_at_k": (sum(hits) / len(top_k)) if top_k else 0.0,
    }


def keyword_recall(context_text: str, expected_keywords: list):
    """Fraction of expected_keywords present anywhere in the retrieved context.
    A cheap, objective proxy for context recall when no gold reference answer exists."""
    if not expected_keywords:
        return None, []
    text = context_text.lower()
    found = [kw for kw in expected_keywords if kw.lower() in text]
    return len(found) / len(expected_keywords), found


def mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0
