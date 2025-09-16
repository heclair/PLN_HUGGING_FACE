# app/main.py
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, ingest, query, chat
from app.services.bootstrap import load_or_seed
from app.services.index import vector_index
from app.core.config import settings

# -------------------------------------------------
# FastAPI + OpenAPI UIs nativas (sem CDN)
# -------------------------------------------------
app = FastAPI(
    title="RAG + Hugging Face (Aula 3)",
    description="Backend organizado com FastAPI, FAISS e Hugging Face (LOCAL/Transformers ou Endpoint).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# -------------------------------------------------
# CORS (para permitir a pasta 'frontend' em outra porta)
# - Você pode sobrescrever com CORS_ORIGINS="http://localhost:8501,http://localhost:3000"
# -------------------------------------------------
_default_origins = [
    "http://localhost:8501", "http://127.0.0.1:8501",  # Streamlit
    "http://localhost:3000", "http://127.0.0.1:3000",  # React/Next
    "http://localhost:5173", "http://127.0.0.1:5173",  # Vite
    "http://localhost:8080", "http://127.0.0.1:8080",  # Vue
]
env_origins = os.getenv("CORS_ORIGINS")
allow_origins = [o.strip() for o in env_origins.split(",")] if env_origins else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Rotas
# -------------------------------------------------
app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(query.router, tags=["query"])
app.include_router(chat.router, tags=["chat"])

# Raiz → Swagger nativo
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Favicon silencioso (evita 404 no navegador)
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)

# -------------------------------------------------
# Ciclo de vida
# -------------------------------------------------
@app.on_event("startup")
def _on_startup():
    # Não travar a UI se o seed falhar
    try:
        total = load_or_seed()
        print(f"[startup] docs carregados: {total}")
    except Exception as e:
        print(f"[startup] load_or_seed falhou: {e}")

@app.on_event("shutdown")
def _on_shutdown():
    # Persistência do índice, se habilitado em settings/.env
    try:
        if getattr(settings, "PERSIST_INDEX", False):
            vector_index.save(settings.INDEX_DIR)
            print(f"[shutdown] índice salvo em {settings.INDEX_DIR}")
    except Exception as e:
        print(f"[shutdown] falha ao salvar índice: {e}")

