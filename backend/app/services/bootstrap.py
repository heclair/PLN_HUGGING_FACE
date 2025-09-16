# app/services/bootstrap.py
from typing import List, Dict, Any
from app.services.index import vector_index
from app.services.embeddings import embeddings_service
from app.core.config import settings

def _add(texts: List[str], metas: List[Dict[str, Any]], chunk: bool = False):
    # se você já tem chunk_text, pode usar; aqui mantemos simples
    vecs = embeddings_service.encode(texts)
    return vector_index.add_documents(texts, metas, vecs)

def seed_with_samples() -> int:
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
    _add(samples, metas, chunk=False)
    return vector_index.count()

def load_or_seed() -> int:
    # 1) tenta carregar de disco
    vector_index.load(settings.INDEX_DIR)
    if vector_index.count() > 0:
        return vector_index.count()
    # 2) se vazio e flag ligada, semeia
    if settings.AUTO_SEED:
        return seed_with_samples()
    return 0
