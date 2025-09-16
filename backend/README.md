## RAG + Hugging Face (FastAPI)

### Rodar
1) python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
2) pip install -r requirements.txt
3) cp .env.example .env  # e edite o token
4) uvicorn app.main:app --reload --port 8000

### Endpoints
- GET /health
- POST /ingest/sample
- POST /ingest/texts
- POST /ingest/file (multipart: .txt)
- POST /query
