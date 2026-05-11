# Conversational SHL Assessment Recommender

This project is a FastAPI backend for recommending SHL assessments through a conversational interface.

## Overview
This API uses a LangGraph-orchestrated stateless RAG (Retrieval-Augmented Generation) pipeline. It handles conversations, detects intent (clarify, recommend, refine, compare), and returns SHL catalog items based on TF-IDF + Keyword boosting. Gemini 2.5 Flash is used exclusively for generating the natural language replies based *only* on the local catalog (`data/shl_catalog_main.json`), ensuring no hallucinated URLs or items.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_genai_api_key
   ```
   *Note: If the key is omitted or the LLM fails, the system will fall back to deterministic response text.*

3. **Data Dependency**
   Ensure that the provided `data/shl_catalog_main.json` file is present in the `data` directory. This is the single source of truth for all recommendations.

## Running Locally

Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

Run the automated test suite with pytest:
```bash
pytest -v
```
This tests schema validation, guardrails, clarifications, and RAG URL grounding.

## API Examples

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I am hiring a Java developer with 4 years experience who works with stakeholders"}]}'
```

## Deployment
This project includes a `render.yaml` for deployment on Render.
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## System Constraints & Behavior
- **Stateless**: Every POST request requires the full `messages` history. The backend reconstructs context from scratch.
- **Catalog-Grounded**: The system strictly recommends items available in the provided JSON catalog. URLs are verified before being returned.
- **Strict Schema**: Response always contains `reply` (str), `recommendations` (list), and `end_of_conversation` (bool).
