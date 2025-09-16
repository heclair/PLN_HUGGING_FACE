from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.utils.chunk import chunk_text
from app.services.embeddings import embeddings_service
from app.services.index import vector_index
from app.models.schemas import IngestTextBody  # ✅ usar schema p/ body JSON

router = APIRouter()

def _ingest_texts_impl(texts: List[str], metas: List[Dict[str, Any]], do_chunk: bool):
    all_chunks, all_metas = [], []
    for i, t in enumerate(texts):
        chunks = chunk_text(t) if do_chunk else [t]
        meta = metas[i] if i < len(metas) else {}
        for c in chunks:
            all_chunks.append(c)
            all_metas.append(meta)
    vecs = embeddings_service.encode(all_chunks)
    return vector_index.add_documents(all_chunks, all_metas, vecs)

# ✅ opção: aceitar GET e POST para facilitar teste no navegador
@router.api_route("/ingest/sample", methods=["GET", "POST"])
def ingest_sample():
    samples = [
        "RAG combina recuperação de informação com geração de texto, melhorando precisão.",
        "Hugging Face Hub oferece modelos, datasets e spaces para IA.",
        "Prompt Engineering é a prática de desenhar prompts para melhorar a resposta de LLMs."
    ]
    metas = [
        {"source": "notas_aula", "topic": "RAG"},
        {"source": "huggingface", "topic": "hub"},
        {"source": "notas_aula", "topic": "prompt_engineering"},
    ]
    return _ingest_texts_impl(samples, metas, do_chunk=False)

# ✅ agora recebe body JSON conforme o schema (fica bonito no Swagger)
@router.post("/ingest/texts")
def ingest_texts(body: IngestTextBody):
    metas = body.metas or [{} for _ in body.texts]
    return _ingest_texts_impl(body.texts, metas, do_chunk=body.chunk)

@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), chunk: bool = Form(True)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Somente .txt neste exemplo.")  # ✅ 400 em vez de JSON solto
    content = (await file.read()).decode("utf-8", errors="ignore")
    return _ingest_texts_impl([content], [{"filename": file.filename}], do_chunk=chunk)
