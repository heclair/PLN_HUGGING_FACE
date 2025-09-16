import os
import json
import requests
from fastapi import HTTPException
from app.core.config import settings

# ------------------------------------------------------------
# Fallback local (Transformers) — usado quando:
# - HF_USE_LOCAL=1 (forçado), OU
# - faltou HF_TOKEN/HF_MODEL, OU
# - a Inference API falhar (404/5xx/rede)
# ------------------------------------------------------------
_LOCAL_PIPE = None
_LOCAL_TASK = None  # guarda a task atual p/ reusar pipeline

def _pick_local_task(model_name: str) -> str:
    """Escolhe a task adequada para o modelo local."""
    name = (model_name or "").lower()
    # Modelos encoder-decoder (T5/FLAN/T0/UL2) usam text2text-generation
    if any(k in name for k in ("t5", "flan", "t0", "ul2")):
        return "text2text-generation"
    return "text-generation"

def _local_generate(prompt: str, temperature: float, max_new_tokens: int) -> str:
    global _LOCAL_PIPE, _LOCAL_TASK
    try:
        from transformers import pipeline
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fallback local indisponível (instale 'transformers' e 'torch'): {e}",
        )

    local_model = getattr(settings, "LOCAL_MODEL", None) or os.getenv("LOCAL_MODEL") or "google/flan-t5-small"
    task = _pick_local_task(local_model)

    try:
        if _LOCAL_PIPE is None or _LOCAL_TASK != task:
            _LOCAL_PIPE = pipeline(task, model=local_model)
            _LOCAL_TASK = task
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao carregar modelo local '{local_model}' (task={task}): {e}",
        )

    try:
        if task == "text-generation":
            out = _LOCAL_PIPE(
                prompt,
                do_sample=True,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
            )[0]["generated_text"]
            # manter comportamento de return_full_text=False
            if out.startswith(prompt):
                out = out[len(prompt):]
            return out.strip()
        else:
            # text2text-generation (Flan/T5) — já retorna só a resposta
            out = _LOCAL_PIPE(
                prompt,
                max_new_tokens=int(max_new_tokens),
            )[0]["generated_text"]
            return out.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar localmente: {e}")

def _should_force_local() -> bool:
    # Flag via settings ou variável de ambiente
    return getattr(settings, "HF_USE_LOCAL", False) or os.getenv("HF_USE_LOCAL") == "1"

# ------------------------------------------------------------
# Chamada principal
# ------------------------------------------------------------
def call_hf_inference(
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    *,
    force_remote: bool = False,  # útil p/ /debug/hf-remote
) -> str:
    # 1) Atalho: usuário forçou local (a menos que force_remote=True)
    if not force_remote and _should_force_local():
        return _local_generate(prompt, temperature, max_new_tokens)

    model_id = (getattr(settings, "HF_MODEL", "") or "").strip()
    token = (getattr(settings, "HF_TOKEN", "") or "").strip()

    # 2) Sem token/modelo: se forçar remoto, erro; senão, local
    if not token or not model_id:
        if force_remote:
            raise HTTPException(status_code=500, detail="HF_TOKEN/HF_MODEL ausentes para chamada remota.")
        return _local_generate(prompt, temperature, max_new_tokens)

    # 3) Tenta Inference API (remoto)
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "return_full_text": False,
            # ajuda a evitar eco do prompt/contexto (nem todo modelo respeita)
            "stop": ["### CONTEXTO", "### PERGUNTA", "### HISTÓRICO", "### RESPOSTA", "```"]
        },
        "options": {"wait_for_model": True},  # aguarda container "acordar"
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if r.status_code != 200:
            # Se der erro remoto: se force_remote=True, propague; senão, caia p/ local
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            detail = f"Inference API retornou {r.status_code} para '{model_id}': {msg}"
            if force_remote:
                raise HTTPException(status_code=502, detail=detail)
            # Fallback local
            try:
                return _local_generate(prompt, temperature, max_new_tokens)
            except HTTPException:
                # se o local também falhar, repasse o erro remoto
                raise HTTPException(status_code=502, detail=detail)

        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()

        if force_remote:
            raise HTTPException(status_code=502, detail=f"Formato inesperado da Inference API: {data}")
        # formato inesperado → tenta local
        return _local_generate(prompt, temperature, max_new_tokens)

    except requests.exceptions.RequestException as e:
        if force_remote:
            raise HTTPException(status_code=502, detail=f"Falha de rede ao chamar a Inference API: {e}")
        # rede falhou → fallback local
        return _local_generate(prompt, temperature, max_new_tokens)
