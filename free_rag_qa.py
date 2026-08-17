import os
import sys
from collections import deque

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIR     = "db/chroma_db_free"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5
MAX_TURNS = 10


def get_groq_key():
    """Get the Groq API key from environment or prompt the user."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    print("\n⚠️  GROQ_API_KEY not found in your .env file.")
    print("   Get a FREE key at: https://console.groq.com/keys")
    print("   Then add this line to your .env file:")
    print("       GROQ_API_KEY=gsk_your_key_here\n")
    key = input("   Or paste your key now (for this session only): ").strip()

    if not key:
        raise ValueError("❌ No Groq API key provided. Cannot continue.")

    os.environ["GROQ_API_KEY"] = key
    return key


def load_vectorstore(embeddings):
    """Load the existing ChromaDB vector store."""
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"\n❌ Database folder '{PERSIST_DIR}' does not exist.\n"
            "   Please run this command first:\n"
            "       python free_rag_ingestion.py"
        )
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_prompt(query: str, context: str, history: str) -> str:
    """Build the RAG prompt sent to the Groq LLM."""
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

User Question: {query}

Answer:"""


def format_history(history_deque: deque) -> str:
    if not history_deque:
        return "(No prior conversation yet.)"

    lines = []
    for msg in history_deque:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_retrieval_query(query: str, history_deque: deque) -> str:
    if not history_deque:
        return query

    recent_user_turns = [m["content"] for m in history_deque if m["role"] == "user"][-3:]
    history_hint = "\n".join(recent_user_turns)
    return f"Previous user context:\n{history_hint}\n\nCurrent question:\n{query}"


def main():
    print("=" * 55)
    print("  Free RAG Q&A System — Groq (Fast & No Quota Issues)")
    print("=" * 55)

    try:
        get_groq_key()
    except ValueError as e:
        print(e)
        return

    print(f"\n[1/3] Loading local HuggingFace embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[2/3] Connecting to ChromaDB at '{PERSIST_DIR}'...")
    try:
        db = load_vectorstore(embeddings)
    except FileNotFoundError as e:
        print(e)
        return

    count = db._collection.count()
    print(f"       Database loaded — {count} indexed chunks ready.")

    print(f"\n[3/3] Connecting to Groq ({GROQ_MODEL})...")
    print("       (Free: 14,400 req/day — keys don't expire!)")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    history = deque(maxlen=MAX_TURNS * 2)

    print("\n✅ System ready!  Type 'quit' or 'exit' to stop.\n")
    print("-" * 55)

    while True:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        retrieval_query = build_retrieval_query(query, history)

        docs = retriever.invoke(retrieval_query)
        if not docs:
            print("⚠️  No matching lectures found in the database.")
            continue

        print(f"\n📚 Top {len(docs)} retrieved lectures:")
        for i, doc in enumerate(docs, 1):
            title = (doc.metadata.get("title") or doc.metadata.get("source_file") or "Untitled source")
            instructor = doc.metadata.get("instructor") or "N/A"
            venue = doc.metadata.get("venue") or doc.metadata.get("source_type") or "Unknown"
            year = doc.metadata.get("year") or "N/A"
            print(f"  [{i}] \"{title}\" — {instructor} ({venue}, {year})")

        context = "\n".join(
            f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
        )

        prompt = build_prompt(query, context, format_history(history))
        print(f"\n🤖 Generating answer with Groq ({GROQ_MODEL})...")
        try:
            chat_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            answer = chat_response.choices[0].message.content
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
            print("\n" + "=" * 55)
            print(answer)
            print("=" * 55)
        except Exception as e:
            print(f"❌ Groq error: {repr(e)}")
            print("   Check your GROQ_API_KEY at: https://console.groq.com/keys")


if __name__ == "__main__":
    main()
