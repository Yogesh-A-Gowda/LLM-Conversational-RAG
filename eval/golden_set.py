"""
Golden test set for evaluating the LectureBank RAG pipeline.

Every in-scope question is grounded in a record actually present in the first
500 rows of LectureBank/alldata.tsv (the MAX_ROWS cutoff used by
free_rag_ingestion.py) or in one of the ingested PDFs, so retrieval hits can
be checked against a known-correct source.

`matchers` describes what counts as a "relevant" retrieved chunk for a
question:
  {"id": "<tsv row id>"}          -> TSV-derived chunk with that metadata id
  {"source_file": "<pdf name>"}   -> chunk extracted from that PDF

`expected_keywords` is used for a deterministic (non-LLM) context-recall
check: the fraction of these terms that show up somewhere in the retrieved
context.
"""

IN_SCOPE_QUESTIONS = [
    {
        "id": "q01",
        "question": "Which lecture covers Long Short Term Memory Networks, and who taught it?",
        "matchers": [{"id": "9"}],
        "expected_keywords": ["Long Short Term Memory", "Radev"],
    },
    {
        "id": "q02",
        "question": "Is there a lecture on WordNet? Which course is it from?",
        "matchers": [{"id": "12"}],
        "expected_keywords": ["Wordnet", "Radev"],
    },
    {
        "id": "q03",
        "question": "What lecture introduces regular expressions for NLP preprocessing?",
        "matchers": [{"id": "24"}],
        "expected_keywords": ["Regular Expressions"],
    },
    {
        "id": "q04",
        "question": "Which lecture introduces language modeling concepts?",
        "matchers": [{"id": "60"}],
        "expected_keywords": ["Language Modeling"],
    },
    {
        "id": "q05",
        "question": "Find the lecture on logistic regression basics.",
        "matchers": [{"id": "90"}],
        "expected_keywords": ["Logistic Regression"],
    },
    {
        "id": "q06",
        "question": "What lecture explains transition-based dependency parsing?",
        "matchers": [{"id": "120"}],
        "expected_keywords": ["Dependency Parsing"],
    },
    {
        "id": "q07",
        "question": "Which lecture from Duke covers adversarial search?",
        "matchers": [{"id": "150"}],
        "expected_keywords": ["Adversarial Search", "Duke"],
    },
    {
        "id": "q08",
        "question": "What lecture gives a probabilistic view of machine learning, from UMD?",
        "matchers": [{"id": "210"}],
        "expected_keywords": ["Probabilistic View", "UMD"],
    },
    {
        "id": "q09",
        "question": "Which lecture compares soft clustering versus hard clustering?",
        "matchers": [{"id": "240"}],
        "expected_keywords": ["Soft Clustering", "Hard Clustering"],
    },
    {
        "id": "q10",
        "question": "Is there an introductory lecture on probability and statistics for NLP/ML?",
        "matchers": [{"id": "270"}],
        "expected_keywords": ["Probability", "Statistics"],
    },
    {
        "id": "q11",
        "question": "Which Carnegie Mellon lecture covers Bayes rule and parameter estimation?",
        "matchers": [{"id": "300"}],
        "expected_keywords": ["Bayes Rule", "Carnegie Mellon"],
    },
    {
        "id": "q12",
        "question": "What lecture from Christopher Manning at Stanford discusses compression?",
        "matchers": [{"id": "360"}],
        "expected_keywords": ["Compression", "Manning"],
    },
    {
        "id": "q13",
        "question": "Which lecture covers frame semantics, and who teaches it?",
        "matchers": [{"id": "490"}],
        "expected_keywords": ["Frame Semantics", "Choi"],
    },
    {
        "id": "q14",
        "question": "Explain the self-attention mechanism used in Transformer models.",
        "matchers": [{"source_file": "cs224n-2024-lecture08-transformers.pdf"}],
        "expected_keywords": ["attention", "query", "key", "value"],
    },
    {
        "id": "q15",
        "question": "What is RLHF and how is it used to align language models?",
        "matchers": [{"source_file": "cs224n-2024-lecture10-instruction-tuning-rlhf.pdf"}],
        "expected_keywords": ["reward", "human feedback", "policy"],
    },
    {
        "id": "q16",
        "question": "How are word embeddings / word vectors learned?",
        "matchers": [{"source_file": "vector25aug.pdf"}],
        "expected_keywords": ["vector", "embedding"],
    },
    {
        "id": "q17",
        "question": "How do recurrent neural networks process sequential data?",
        "matchers": [{"source_file": "rnnjan25.pdf"}],
        "expected_keywords": ["recurrent", "sequence", "hidden state"],
    },
    {
        "id": "q18",
        "question": "Explain how the Naive Bayes classifier is used for text classification.",
        "matchers": [{"source_file": "nb24aug.pdf"}],
        "expected_keywords": ["Naive Bayes", "probability", "classification"],
    },
]

OUT_OF_SCOPE_QUESTIONS = [
    {"id": "oos01", "question": "What is the capital of France?"},
    {"id": "oos02", "question": "Give me a recipe for chocolate chip cookies."},
    {"id": "oos03", "question": "What is today's stock price of Apple Inc.?"},
    {"id": "oos04", "question": "Who won the most recent Super Bowl?"},
    {"id": "oos05", "question": "How do I fix a flat bike tire?"},
]
