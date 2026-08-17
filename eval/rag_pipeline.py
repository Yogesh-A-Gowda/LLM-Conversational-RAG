"""
Thin wrapper around the exact retrieval/prompt logic in app.py, so the eval
scripts exercise the same code path the live app uses instead of a
reimplementation that could silently drift from it.

app.py's globals (_embeddings/_db) are only populated by its FastAPI
`startup` event, which never fires here since we import rather than run it
under uvicorn -- so this module does its own (identical) setup and calls
app.py's pure helper functions directly.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import requests

import app as rag_app  # the project's app.py

_embeddings = None
_db = None
_groq_client = None


def get_db():
    global _embeddings, _db
    if _db is None:
        _embeddings = HuggingFaceEmbeddings(model_name=rag_app.EMBEDDING_MODEL)
        _db = Chroma(
            persist_directory=rag_app.CHROMA_DB_PATH,
            embedding_function=_embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _db


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def retrieve_with_scores(question: str, k: int = None):
    """Same call app.py's /api/chat makes: similarity_search_with_relevance_scores,
    falling back to distance->relevance conversion. Returns (docs_with_scores, elapsed_ms)."""
    db = get_db()
    k = k or rag_app.TOP_K
    t0 = time.perf_counter()
    try:
        result = db.similarity_search_with_relevance_scores(question, k=k)
    except AttributeError:
        docs_with_distance = db.similarity_search_with_score(question, k=k)
        result = []
        for doc, distance in docs_with_distance:
            try:
                relevance = 1.0 / (1.0 + float(distance))
            except Exception:
                relevance = 0.0
            result.append((doc, relevance))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def build_context(docs) -> str:
    return "\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


def generate_groq(prompt: str, model: str = "llama-3.3-70b-versatile"):
    client = get_groq_client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return resp.choices[0].message.content, elapsed_ms


def generate_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 180):
    t0 = time.perf_counter()
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json()["message"]["content"], elapsed_ms


def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


build_prompt = rag_app.build_prompt
detect_scope = rag_app.detect_scope
MIN_SCOPE_RELEVANCE = rag_app.MIN_SCOPE_RELEVANCE
TOP_K = rag_app.TOP_K
