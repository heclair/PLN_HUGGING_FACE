from fastapi import APIRouter
from app.models.schemas import QueryBody
from app.services.rag import answer_with_rag

router = APIRouter()

@router.post("/query")
def query_rag(body: QueryBody):
    return answer_with_rag(
        question=body.question,
        k=body.top_k,
        temperature=body.temperature,
        max_new_tokens=body.max_new_tokens,
    )
