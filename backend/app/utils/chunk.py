from typing import List

def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> List[str]:
    """Split simples por palavras (heurístico)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks if chunks else [text]
