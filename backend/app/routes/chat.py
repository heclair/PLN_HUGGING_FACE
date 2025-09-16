from fastapi import APIRouter
from app.models.schemas import ChatBody
from app.services.chat_memory import chat_memory
from app.services.rag import chat_answer

router = APIRouter()

@router.post("/chat")
def chat(body: ChatBody):
    # usa o histórico enviado OU o salvo no servidor
    server_hist = chat_memory.get(body.session_id)
    history = body.history if body.history else server_hist

    result = chat_answer(
        message=body.message,
        history=history,
        top_k=body.top_k,
        temperature=body.temperature,
        max_new_tokens=body.max_new_tokens,
        system_prompt=body.system_prompt,
    )

    # atualiza memória do servidor
    chat_memory.append(body.session_id, "user", body.message)
    chat_memory.append(body.session_id, "assistant", result["answer"])

    result["session_id"] = body.session_id
    result["history_len"] = len(chat_memory.get(body.session_id))
    return result

@router.post("/chat/reset/{session_id}")
def chat_reset(session_id: str):
    chat_memory.reset(session_id)
    return {"ok": True, "session_id": session_id}
