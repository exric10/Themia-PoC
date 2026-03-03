# Themia – Legal AI Copilot · PoC

Local demo of a legal RAG + agent pipeline. Runs entirely on your machine — no API keys required.

## Stack

- **Ollama** – runs `mistral` locally for LLM inference
- **FastAPI** – backend with the full 6-step RAG + agent pipeline
- **HTML/nginx** – frontend UI with pipeline visualisation

## Quickstart

```bash
docker compose up --build
```

> First run downloads the `mistral` model (~4 GB). Takes 2–5 min.
> Watch for `mistral pulled successfully` in the logs.

Then open **http://localhost:3000**

```bash
docker compose down      # stop (model stays downloaded)
docker compose down -v   # stop + delete model data
```

## Requirements

- Docker Desktop (Mac/Windows) or Docker + Compose (Linux)
- ~5 GB disk space
- 8 GB RAM minimum (16 GB recommended)

## Project Structure

```
themia/
├── docker-compose.yml
├── backend/
│   ├── main.py          # FastAPI + RAG pipeline
│   └── Dockerfile
└── frontend/
    ├── index.html       # UI
    ├── nginx.conf
    └── Dockerfile
```

## Agent Pipeline

| Step | What it does |
|------|-------------|
| 1 · Intent Classification | Detects domain, jurisdiction, risk level, and which corpora to search |
| 2 · Query Expansion | Generates synonym variants to improve retrieval recall |
| 3 · Hybrid Retrieval | BM25 + dense (TF-IDF) merged via Reciprocal Rank Fusion |
| 4 · Reranking + Dedup | Cross-encoder joint scoring, removes exact and near-duplicate chunks |
| 5 · Guided Reasoning | CoT prompt → Ollama → citation-anchored answer |
| 6 · Citation Verification | Checks index bounds, uncited legal claims, stale sources |

## Swap the Model

Change `OLLAMA_MODEL` in `backend/main.py` and the pull command in `docker-compose.yml`.

| Model | Size | RAM |
|-------|------|-----|
| `mistral` | 4 GB | 8 GB |
| `llama3.2` | 2 GB | 6 GB |
| `llama3.1:8b` | 5 GB | 10 GB |
```
