import os
import uuid
from collections import deque
import requests
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

CHROMA_DB_PATH  = os.getenv(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db_free"),
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 5
MAX_TURNS       = 10
MIN_SCOPE_RELEVANCE = 0.3

app = FastAPI(title="LectureBank RAG API")

_embeddings = None
_db         = None
_conversations = {}


@app.on_event("startup")
async def startup():
    global _embeddings, _db
    import asyncio
    import threading
    
    def load_models():
        global _embeddings, _db
        try:
            print("[APP] Loading HuggingFace embedding model...")
            _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            print(f"[APP] Connecting to ChromaDB at: {CHROMA_DB_PATH}")
            _db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=_embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            count = _db._collection.count()
            print(f"[APP] Ready - {count} lecture chunks indexed.")
        except Exception as e:
            print(f"[APP] ERROR during startup: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background thread to avoid blocking startup
    thread = threading.Thread(target=load_models, daemon=True)
    thread.start()
    print("[APP] Server starting (DB loading in background)...")


class ChatRequest(BaseModel):
    question:     str
    conversation_id: str | None = None
    backend:      str   = "groq"
    groq_model:   str   = "openai/gpt-oss-120b"
    ollama_model: str   = "llama3.1:8b"


class Source(BaseModel):
    title:      str
    instructor: str
    venue:      str
    year:       str
    url:        str


class ChatResponse(BaseModel):
    conversation_id: str
    answer:  str
    sources: list[Source]
    out_of_scope: bool = False
    warning: str | None = None


def build_prompt(question: str, context: str, history: str) -> str:
    return f"""You are an expert educational assistant. Answer the user's question based on the lecture materials provided.

Instructions:
- Synthesize a comprehensive answer from the retrieved lecture content
- Be specific and cite which lecture(s) the information comes from
- If the exact answer isn't in the materials, say what's available and suggest related concepts
- Provide practical examples or explanations when relevant

Recent conversation context (most recent last):
{history}

Retrieved Lecture Content:
{context}

User Question: {question}

Answer:"""


def retrieve_docs(question: str):
    retriever = _db.as_retriever(search_kwargs={"k": TOP_K})
    return retriever.invoke(question)


def retrieve_docs_with_scores(question: str):
    if _db is None:
        raise HTTPException(status_code=503, detail="Vector database is not ready yet. Please try again in a moment.")

    try:
        return _db.similarity_search_with_relevance_scores(question, k=TOP_K)
    except AttributeError:
        docs_with_distance = _db.similarity_search_with_score(question, k=TOP_K)
        converted = []
        for doc, distance in docs_with_distance:
            try:
                relevance = 1.0 / (1.0 + float(distance))
            except Exception:
                relevance = 0.0
            converted.append((doc, relevance))
        return converted


def docs_to_sources(docs) -> list[Source]:
    def clean(value, default=""):
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    return [
        Source(
            title=clean(d.metadata.get("title"), clean(d.metadata.get("source_file"), "Untitled source")),
            instructor=clean(d.metadata.get("instructor"), "N/A"),
            venue=clean(d.metadata.get("venue"), clean(d.metadata.get("source_type"), "Unknown")),
            year=clean(d.metadata.get("year"), "N/A"),
            url=clean(d.metadata.get("url"), ""),
        )
        for d in docs
    ]


def get_or_create_conversation(conversation_id: str | None):
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]

    new_id = conversation_id or str(uuid.uuid4())
    _conversations[new_id] = deque(maxlen=MAX_TURNS * 2)
    return new_id, _conversations[new_id]


def format_history(history_deque: deque) -> str:
    if not history_deque:
        return "(No prior conversation yet.)"

    formatted = []
    for msg in history_deque:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def build_retrieval_query(question: str, history_deque: deque) -> str:
    if not history_deque:
        return question

    recent_user_turns = [m["content"] for m in history_deque if m["role"] == "user"][-3:]
    history_hint = "\n".join(recent_user_turns)
    return f"Previous user context:\n{history_hint}\n\nCurrent question:\n{question}"


def detect_scope(retrieved_with_scores) -> bool:
    if not retrieved_with_scores:
        return False

    top_score = max(score for _, score in retrieved_with_scores)
    return top_score >= MIN_SCOPE_RELEVANCE


@app.get("/api/status")
def status():
    """Check which backends are available."""
    groq_available = bool(os.getenv("GROQ_API_KEY", "").strip())

    ollama_available = False
    ollama_models = []
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            ollama_available = True
            ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    db_count = _db._collection.count() if _db else 0

    return {
        "groq":          groq_available,
        "ollama":        ollama_available,
        "ollama_models": ollama_models,
        "db_chunks":     db_count,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run RAG: retrieve relevant lectures then generate an answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    conversation_id, history_deque = get_or_create_conversation(req.conversation_id)
    history_text = format_history(history_deque)
    retrieval_query = build_retrieval_query(req.question, history_deque)

    docs_with_scores = retrieve_docs_with_scores(retrieval_query)
    docs = [doc for doc, _ in docs_with_scores]
    in_scope = detect_scope(docs_with_scores)
    sources = docs_to_sources(docs)
    context = "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    prompt  = build_prompt(req.question, context, history_text)

    warning = None
    if not in_scope:
        warning = (
            "This question looks outside the scope of this RAG dataset. "
            "I can answer questions grounded in LectureBank NLP lectures (topics, instructors, concepts, methods, and lecture content)."
        )
        answer = (
            "I could not find enough relevant lecture context for that question. "
            "Please ask about NLP lecture content in this dataset, for example: parsing, embeddings, language models, machine translation, or specific instructors/universities."
        )
        history_deque.append({"role": "user", "content": req.question})
        history_deque.append({"role": "assistant", "content": answer})
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            sources=[],
            out_of_scope=True,
            warning=warning,
        )

    answer = ""
    if req.backend == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY not set in .env")
        try:
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=req.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            err_text = str(e)
            if "invalid_api_key" in err_text.lower() or "error code: 401" in err_text.lower():
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Groq authentication failed (invalid API key). "
                        "Update GROQ_API_KEY in .env and restart the API server."
                    ),
                )
            raise HTTPException(status_code=500, detail=f"Groq error: {err_text}")

    elif req.backend == "ollama":
        try:
            model_list_resp = requests.get("http://localhost:11434/api/tags", timeout=4)
            model_list_resp.raise_for_status()
            available_models = [m.get("name", "") for m in model_list_resp.json().get("models", [])]
            if req.ollama_model not in available_models:
                available_text = ", ".join(available_models[:8]) or "(none found)"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Ollama model '{req.ollama_model}' is not installed. "
                        f"Available models: {available_text}. "
                        "Run: ollama pull <model>"
                    ),
                )

            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model":    req.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                },
                timeout=180,
            )
            if resp.status_code >= 400:
                error_msg = ""
                try:
                    body = resp.json()
                    error_msg = body.get("error") or body.get("message") or ""
                except Exception:
                    error_msg = resp.text

                if (
                    "GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)" in error_msg
                    or "0xc0000409" in error_msg
                    or "stack-based buffer" in error_msg.lower()
                ):
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Ollama model process crashed (known runtime issue on this setup). "
                            "Use a stable local model such as 'mistral:latest' or 'llama3.1:8b', "
                            "or update Ollama and retry this model."
                        ),
                    )

                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Ollama error ({resp.status_code}): {error_msg[:500]}",
                )
            answer = resp.json()["message"]["content"]
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail="Ollama model load timed out. This usually happens on the first query because "
                       "your computer is loading the model weights into RAM. The model should now be ready in memory — "
                       "please wait a few seconds and try submitting your question again!"
            )
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama at http://localhost:11434. Start Ollama and try again.",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {req.backend}")

    history_deque.append({"role": "user", "content": req.question})
    history_deque.append({"role": "assistant", "content": answer})

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        out_of_scope=False,
        warning=None,
    )


@app.post("/api/conversation/reset")
def reset_conversation(payload: dict):
    conversation_id = (payload or {}).get("conversation_id", "")
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required.")
    _conversations.pop(conversation_id, None)
    return {"ok": True}


_current_dir = os.path.dirname(os.path.abspath(__file__))
_static_dir = os.path.join(_current_dir, "static")

@app.get("/")
def root():
    """Serve the main UI."""
    return FileResponse(os.path.join(_static_dir, "index.html"), media_type="text/html")

# Mount static files at /static/* path (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
