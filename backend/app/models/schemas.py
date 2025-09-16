from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from typing import List, Optional
from pydantic import BaseModel

class IngestTextBody(BaseModel):
    texts: List[str]
    metas: Optional[List[Dict[str, Any]]] = None
    chunk: bool = True

class QueryBody(BaseModel):
    question: str
    top_k: int = 3
    temperature: float = 0.7
    max_new_tokens: int = 256

    # ⬇ isto faz o Swagger já vir preenchido com um exemplo válido
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "O que é RAG e por que é útil?",
                "top_k": 3,
                "temperature": 0.2,
                "max_new_tokens": 128
            }
        }
    }
    
class ChatMessage(BaseModel):
    role: str            # "user" | "assistant"
    content: str

class ChatBody(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = []  # opcional: cliente pode enviar histórico
    top_k: int = 3
    temperature: float = 0.7
    max_new_tokens: int = 256
    system_prompt: Optional[str] = None    
