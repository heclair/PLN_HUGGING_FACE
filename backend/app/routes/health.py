# app/routes/health.py
from fastapi import APIRouter
from app.services.index import vector_index
from app.core.config import settings
from app.core.llm import call_hf_inference

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "docs": vector_index.count(),
        "index_built": vector_index.index is not None,
    }

@router.get("/debug/config")
def debug_config():
    return {
        "HF_MODEL": settings.HF_MODEL,
        "HF_MODEL_repr": repr(settings.HF_MODEL),
        "HF_TOKEN_set": bool(settings.HF_TOKEN),
        "EMBED_MODEL": settings.EMBED_MODEL,
        "HF_USE_LOCAL": bool(getattr(settings, "HF_USE_LOCAL", False)),
        "LOCAL_MODEL": getattr(settings, "LOCAL_MODEL", "google/flan-t5-small"),
    }

# ✅ testa via call_hf_inference (pode usar remoto ou fallback local, conforme .env)
@router.get("/debug/hf")
def debug_hf():
    out = call_hf_inference("Diga 'ok' e nada mais.", temperature=0.1, max_new_tokens=5)
    return {"hf_ok": True, "sample": out}

# ✅ testa a Inference API REMOTA obrigatoriamente (sem fallback local)
@router.get("/debug/hf-remote")
def debug_hf_remote():
    out = call_hf_inference(
        "Diga 'ok' e nada mais.", temperature=0.1, max_new_tokens=5, force_remote=True
    )
    return {"hf_remote_ok": True, "sample": out}
