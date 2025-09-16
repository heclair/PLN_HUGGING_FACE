# app/services/index.py
from __future__ import annotations
from typing import List, Dict, Any
import os
import json
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    raise RuntimeError(
        "FAISS não está instalado. Rode: pip install faiss-cpu\n"
        f"Erro original: {e}"
    )

class VectorIndex:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.docs: List[Dict[str, Any]] = []
        self.dim: int | None = None

    # ---------- util ----------
    @staticmethod
    def _as_ndarray(vectors) -> np.ndarray:
        if isinstance(vectors, np.ndarray):
            arr = vectors
        else:
            arr = np.array(vectors, dtype=np.float32)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        # normaliza linha a linha para usar cosine via Inner Product
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    # ---------- API ----------
    def count(self) -> int:
        """Total de documentos carregados no índice."""
        return len(self.docs)

    def add_documents(
        self,
        texts: List[str],
        metas: List[Dict[str, Any]],
        vectors,
    ) -> Dict[str, Any]:
        """
        Adiciona documentos e vetores ao índice.
        - texts: lista de strings
        - metas: lista de metadados (mesmo comprimento de texts, ou será preenchido com {})
        - vectors: array (n, dim) em float32
        """
        if not texts:
            return {"ingested": 0, "total_docs": self.count()}

        # garantir metas do mesmo tamanho
        if len(metas) < len(texts):
            metas = metas + [{} for _ in range(len(texts) - len(metas))]

        vecs = self._as_ndarray(vectors)
        if vecs.ndim != 2:
            raise ValueError(f"Esperado shape (n, dim) para vectors, obtido {vecs.shape}")

        # cria índice se necessário
        if self.index is None:
            self.dim = int(vecs.shape[1])
            # usaremos Inner Product com vetores L2-normalizados (equivale a cosine)
            self.index = faiss.IndexFlatIP(self.dim)

        # valida dimensão
        if self.dim is None or self.dim != vecs.shape[1]:
            raise ValueError(f"Dimensão dos vetores ({vecs.shape[1]}) difere do índice ({self.dim}).")

        # normaliza e adiciona
        vecs = self._l2_normalize(vecs)
        self.index.add(vecs)

        start_id = self.count()
        for i, t in enumerate(texts):
            self.docs.append({
                "id": start_id + i,
                "text": t,
                "meta": metas[i] if i < len(metas) else {},
            })

        return {"ingested": len(texts), "total_docs": self.count()}

    def search(self, query_vectors, k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca os top-k documentos mais similares ao primeiro vetor de consulta.
        Retorna lista de dicts: {"id", "text", "meta"}.
        """
        if self.index is None or self.count() == 0:
            return []

        q = self._as_ndarray(query_vectors)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Dimensão do vetor de consulta ({q.shape[1]}) difere do índice ({self.dim}).")

        q = self._l2_normalize(q)
        k = max(1, min(k, self.count()))
        distances, indices = self.index.search(q, k)
        top_ids = [int(i) for i in indices[0] if i != -1]

        # monta resposta
        results: List[Dict[str, Any]] = []
        for did in top_ids:
            d = self.docs[did]
            results.append({"id": d["id"], "text": d["text"], "meta": d.get("meta", {})})
        return results

    # ---------- persistência ----------
    def save(self, path: str = "data") -> None:
        os.makedirs(path, exist_ok=True)
        # índice
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        # docs
        with open(os.path.join(path, "docs.jsonl"), "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def load(self, path: str = "data") -> None:
        idx_path = os.path.join(path, "faiss.index")
        docs_path = os.path.join(path, "docs.jsonl")
        if os.path.exists(idx_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(idx_path)
            self.dim = int(self.index.d)
            with open(docs_path, "r", encoding="utf-8") as f:
                self.docs = [json.loads(line) for line in f]
        else:
            # mantém vazio
            self.index = None
            self.docs = []
            self.dim = None
            
def search_with_scores(self, query_vectors, k: int = 3):
    if self.index is None or self.count() == 0:
        return []
    q = self._as_ndarray(query_vectors)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.shape[1] != self.dim:
        raise ValueError(f"Dimensão do vetor de consulta ({q.shape[1]}) difere do índice ({self.dim}).")
    q = self._l2_normalize(q)
    k = max(1, min(k, self.count()))
    distances, indices = self.index.search(q, k)  # IP em vetores normalizados ≈ cos
    out = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        d = self.docs[int(idx)]
        score = float(distances[0][rank])  # ∈ [-1, 1]
        out.append({"id": d["id"], "text": d["text"], "meta": d.get("meta", {}), "score": score})
    return out
           

# singleton exportado
vector_index = VectorIndex()
