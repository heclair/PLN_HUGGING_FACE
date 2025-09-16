# app/services/embeddings.py
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingsService:
    def __init__(self):
        # carrega o modelo multilíngue
        self.model = SentenceTransformer(settings.EMBED_MODEL)
    
    @staticmethod
    def _normalize_text(s: str) -> str:
        # NFKC + casefold + colapsa espaços => robusto p/ maiúsculas/minúsculas/acentos
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKC", s).casefold()
        s = " ".join(s.split())
        return s

    def encode(self, texts):
        if not texts:
            return np.zeros((0, 384), dtype="float32")  # tamanho padrão p/ MiniLM; o lib ajusta conforme o modelo
        normed = [self._normalize_text(t) for t in texts]
        vecs = self.model.encode(normed, normalize_embeddings=True)  # L2-normaliza para busca por dot-product
        return np.asarray(vecs, dtype="float32")

embeddings_service = EmbeddingsService()
