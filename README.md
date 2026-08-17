# RAG Conversational AI

LectureBank-based retrieval-augmented question answering application with a FastAPI backend, browser UI, local Chroma vector database, Hugging Face embeddings, and Groq or Ollama generation.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Git
- Approximately 2 GB of free disk space for Python packages, the embedding model, and the local vector database
- A Groq API key for Groq generation, or Ollama for local generation

Python packages are listed in `requirements.txt`. The data-ingestion packages are listed in `requirements-data.txt`.

## Project Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-data.txt
```

On macOS or Linux, activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-data.txt
```

## Environment Variables

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
```

The Groq key is required only when using the Groq backend or Groq-based evaluations. Keep `.env` private; it is ignored by Git.

## Data and Vector Database

The application reads LectureBank data from `LectureBank/` and uses the existing local vector database at `db/chroma_db_free/`.

To rebuild the vector database after adding or changing LectureBank data:

```powershell
python free_rag_ingestion.py
```

The ingestion script skips rebuilding when the database already exists. Delete `db/chroma_db_free/` first if a full rebuild is required.

## Run the Application

Start the API and web application:

```powershell
python app.py
```

Open http://localhost:8000 in a browser. The API status endpoint is available at http://localhost:8000/api/status.

For the command-line Groq client instead:

```powershell
python free_rag_qa.py
```

## Ollama Backend

Install Ollama, start it, and download a model, for example:

```powershell
ollama pull llama3.1:8b
```

Select Ollama in the web interface. Ollama must be running at `http://localhost:11434`.

## Evaluation

The evaluation scripts are in `eval/`. Run individual evaluations from the repository root:

```powershell
python -m eval.run_retrieval_eval
python -m eval.run_generation_eval --limit 8
python -m eval.run_latency_bench --limit 8
```

Run the complete suite:

```powershell
python -m eval.run_all --gen-limit 8 --latency-limit 8
```

Evaluation scripts create `eval/results/` automatically when needed. The directory is kept in the repository with `.gitkeep`, while generated JSON reports and HTML dashboards are ignored by Git.

Generation and latency evaluations may call Groq and can take longer than retrieval evaluation. Latency evaluation also compares Ollama when it is available.

## Evaluation Dashboards

The generated HTML dashboards live under `eval/results/` and can be opened in a browser after the evaluation runs.

### Groq dashboard

```powershell
python -m eval.run_all --gen-limit 8 --latency-limit 8
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/eval/results/results_dashboard.html
```

### Ollama dashboard

```powershell
ollama serve
python -m eval.run_generation_eval_ollama --limit 8
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/eval/results/results_dashboard_ollama.html
```

These dashboards visualize generation quality, retrieval quality, and latency benchmark results in a presentation-friendly format.

## Repository Contents

- `app.py`: FastAPI server and browser application API
- `static/index.html`: browser interface
- `free_rag_ingestion.py`: LectureBank ingestion and Chroma database creation
- `free_rag_qa.py`: command-line question-answering client
- `eval/`: retrieval, generation, and latency evaluation code
- `eval/results/`: generated evaluation output directory
- `LectureBank/`: local dataset directory; only its README is tracked
- `db/`: local vector database; ignored by Git
