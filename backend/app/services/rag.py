from typing import List, Dict, Any, Optional
import re, difflib, unicodedata

from app.services.embeddings import embeddings_service
from app.services.index import vector_index
from app.core.llm import call_hf_inference

# Limiar mínimo de similaridade (cosine) para aceitar um contexto
MIN_SIM = 0.18

# ---------------------------
# utils
# ---------------------------
_STOPWORDS_PT = {
    "a","o","as","os","um","uma","uns","umas","de","do","da","dos","das","em","no","na","nos","nas",
    "para","por","e","ou","que","com","se","ao","aos","à","às","é","são","como","sobre","até","mais",
    "menos","sem","sua","seu","suas","seus","minha","meu","nossa","nosso","nossas","nossos","isto",
    "isso","aquilo","este","esta","esse","essa","aquele","aquela","ele","ela","eles","elas","você",
    "vocês","eu","me","te","se","lhe","nos","vos","del","das","dos"
}

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def _tokenize(text: str) -> List[str]:
    t = _normalize(text)
    toks = re.findall(r"[a-z0-9]+", t)
    toks = [w for w in toks if len(w) >= 3 and w not in _STOPWORDS_PT]
    # sinônimos simples
    if "h2o" in toks and "agua" not in toks:
        toks.append("agua")
    return toks

def _keyword_overlap_count(question_tokens: List[str], text: str) -> int:
    doc_tokens = set(_tokenize(text))
    qset = set(question_tokens)
    return len(doc_tokens & qset)

def _filter_by_threshold(hits: List[Dict[str, Any]], min_sim: float = MIN_SIM) -> List[Dict[str, Any]]:
    """Aceita itens com 'orig_score' ou 'score' >= limiar."""
    if not hits:
        return []
    key = "orig_score" if "orig_score" in hits[0] else ("score" if "score" in hits[0] else None)
    if key is None:
        return hits
    return [h for h in hits if float(h.get(key, 0.0)) >= min_sim]

def _cleanup_answer(txt: str) -> str:
    """Remove ecos do prompt/contexto, bullets e mantém só a 1ª frase razoável."""
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r"\[?Doc\s*\d+\]?:?.*\n?", "", txt)
    txt = re.sub(r"(HISTÓRICO:|CONTEXTO:|PERGUNTA|RESPOSTA:|###|\`\`\`)", "", txt, flags=re.I)
    txt = re.sub(r"^\s*[-•*>\u2022]+\s*", "", txt)       # bullets no início
    txt = re.sub(r"^\s*[\(\[\{\“\"'`]+", "", txt)        # abre parêntese/aspas no início
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\!\?])\s+", txt)
    first = (parts[0] if parts and parts[0] else txt).strip()
    if len(first) > 300:
        first = first[:300].rstrip() + "…"
    first = re.sub(r"^[-•*>\u2022]+\s*", "", first).lstrip("([{\"'` ").strip()
    return first

def _looks_bad(txt: str) -> bool:
    """Heurística de qualidade: muito curto, sem letras, ou começa “errado”."""
    if not txt:
        return True
    if len(txt) < 15:
        return True
    if not re.search(r"[A-Za-zÀ-ÿ]", txt):
        return True
    if re.match(r"^\s*[-•*>\u2022\(\[\{]", txt):
        return True
    return False

def _too_similar_to_question(answer: str, question: str) -> bool:
    """Detecta eco da pergunta na resposta."""
    a = _normalize(answer)
    q = _normalize(question)
    ratio = difflib.SequenceMatcher(None, a, q).ratio()
    return a.startswith(q[:20]) or ratio >= 0.80

def _synthesize_from_context_general(contexts: List[Dict[str, Any]]) -> str:
    """
    Fallback determinístico: pega a 1ª frase do 1º doc (ou um recorte curto)
    e finaliza com (Fontes: ...). Funciona para qualquer tema (ex.: Água).
    """
    if not contexts:
        return "Não sei com base nos documentos disponíveis."
    t = (contexts[0].get("text") or "").strip()
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    sent = (parts[0] if parts and parts[0] else t).strip()
    if len(sent) < 10:
        sent = t[:160].strip()
    if len(sent) > 220:
        sent = sent[:220].rstrip() + "…"
    fontes = ", ".join([f"Doc {c['id']}" for c in contexts])
    return f"{sent} (Fontes: {fontes})"

def _hybrid_rerank(hits: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    """
    Re-ranqueia combinando score do embedding + sobreposição de palavras da pergunta.
    Dá bônus de até +0.35 para docs que contêm termos da pergunta.
    """
    q_tokens = _tokenize(question)
    if not hits:
        return []

    updated = []
    for h in hits:
        base = float(h.get("score", 0.0))
        text = h.get("text", "") or ""
        overlap = _keyword_overlap_count(q_tokens, text)  # 0,1,2,...
        # bônus de palavras: até 0.35 (3+ overlaps saturam)
        bonus = 0.35 * min(1.0, overlap / 3.0)
        new_score = base + bonus
        h2 = {**h, "orig_score": base, "score": new_score, "_overlap": int(overlap)}
        updated.append(h2)

    # ordena por score híbrido desc, depois por score original desc
    updated.sort(key=lambda x: (x.get("score", 0.0), x.get("orig_score", 0.0)), reverse=True)

    # Se existir pelo menos um com overlap>0, descartamos os que têm 0 overlap
    if any(h.get("_overlap", 0) > 0 for h in updated):
        updated = [h for h in updated if h.get("_overlap", 0) > 0]

    return updated

def _retrieve_contexts(question: str, k: int) -> List[Dict[str, Any]]:
    q_vec = embeddings_service.encode([question])
    if hasattr(vector_index, "search_with_scores"):
        hits = vector_index.search_with_scores(q_vec, k=k)
    else:
        hits = vector_index.search(q_vec, k=k)
        # garante campo score mesmo sem faiss score exposto
        hits = [{**h, "score": 1.0} for h in hits]

    hits = _hybrid_rerank(hits, question)
    return _filter_by_threshold(hits, MIN_SIM)

# ---------------------------
# RAG "clássico"
# ---------------------------
def make_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join([f"[Doc {c['id']}] {c['text']}" for c in contexts]) if contexts else "(sem contexto)"
    # lista de palavras da pergunta para orientar o modelo
    kw = ", ".join(sorted(set(_tokenize(question)))) or "(nenhuma)"
    sys = (
        "Você é um assistente útil e conciso. Responda SOMENTE com base no CONTEXTO a seguir.\n"
        "Priorize passagens que contenham as palavras da pergunta. Se houver discrepância, ignore passagens fora do tema.\n"
        f"Palavras da pergunta: {kw}\n"
        "Se a resposta não estiver no contexto, diga 'Não sei com base nos documentos disponíveis.'.\n"
        "Ao final, cite os IDs das fontes usadas no formato (Fontes: Doc X, Doc Y).\n"
        "Formato: uma única frase, sem iniciar com traços/bullets e sem copiar o CONTEXTO."
    )
    user = f"CONTEXTO:\n{context_block}\n\nPERGUNTA: {question}\nRESPOSTA:"
    return f"{sys}\n{user}"

def top_k_contexts(question: str, k: int = 3) -> List[Dict[str, Any]]:
    return _retrieve_contexts(question, k)

def answer_with_rag(
    question: str,
    k: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 256
):
    ctx = top_k_contexts(question, k=k)
    if not ctx:  # ⚠️ sem contexto relevante → não chama LLM
        return {
            "answer": "Não sei com base nos documentos disponíveis.",
            "sources": [],
            "meta": [],
            "debug": {"prompt": "(sem contexto)"},
        }

    prompt = make_prompt(question, ctx)
    llm_answer = call_hf_inference(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    clean = _cleanup_answer(llm_answer)

    # Anti-eco / qualidade ruim → sintetiza a partir do contexto
    if _looks_bad(clean) or _too_similar_to_question(clean, question):
        clean = _synthesize_from_context_general(ctx)
    else:
        if "(Fontes:" not in clean:
            fontes = ", ".join([f"Doc {c['id']}" for c in ctx])
            clean = f"{clean} (Fontes: {fontes})"

    return {
        "answer": clean,
        "sources": [c["id"] for c in ctx],
        "meta": [c.get("meta", {}) for c in ctx],
        "debug": {"prompt": prompt[:1000]},
    }

# ---------------------------
# CHAT (histórico)
# ---------------------------
def make_chat_prompt(
    question: str,
    contexts: List[Dict[str, Any]],
    history: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> str:
    context_block = "\n".join([f"- (Doc {c['id']}): {c['text']}" for c in contexts]) if contexts else "(sem contexto)"
    kw = ", ".join(sorted(set(_tokenize(question)))) or "(nenhuma)"
    sys = system_prompt or (
        "Você é um assistente técnico e objetivo. REGRAS:\n"
        "1) Responda em 1 frase, em português, sem listar nem copiar trechos do CONTEXTO.\n"
        "2) Não use marcadores (ex.: '-', '*', '•') nem parênteses de abertura no início da resposta.\n"
        "3) Não repita nem reformule a pergunta; não inicie com 'Explique'/'Resuma'.\n"
        "4) Responda SOMENTE com base no CONTEXTO (e no histórico, se útil). Se faltar informação, diga: 'Não sei com base nos documentos disponíveis.'\n"
        "5) Priorize passagens que contenham as palavras da pergunta; ignore passagens fora do tema.\n"
        "6) Termine com as fontes no formato (Fontes: Doc X, Doc Y)."
    )
    # usa pouco histórico para reduzir viés/eco
    hist_lines = []
    for h in history[-4:]:
        role = "Usuário" if h.get("role") == "user" else "Assistente"
        hist_lines.append(f"{role}: {h.get('content','')}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sem histórico)"

    return (
        f"{sys}\n\n"
        f"### HISTÓRICO\n{hist_block}\n\n"
        f"### PALAVRAS DA PERGUNTA\n{kw}\n\n"
        f"### CONTEXTO (use APENAS como base; NÃO copie)\n```\n{context_block}\n```\n\n"
        f"### PERGUNTA\n{question}\n\n"
        f"### RESPOSTA (1 frase, sem bullets, com fontes):"
    )

def chat_answer(
    message: str,
    history: List[Dict[str, str]],
    top_k: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = _retrieve_contexts(message, top_k)

    if not ctx:
        return {
            "answer": "Não sei com base nos documentos disponíveis.",
            "sources": [],
            "meta": [],
            "debug": {"prompt": "(sem contexto)"},
        }

    prompt = make_chat_prompt(message, ctx, history, system_prompt=system_prompt)
    llm_answer = call_hf_inference(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    clean = _cleanup_answer(llm_answer)

    # Anti-eco / qualidade ruim
    if _looks_bad(clean) or _too_similar_to_question(clean, message):
        clean = _synthesize_from_context_general(ctx)
    else:
        if "(Fontes:" not in clean:
            fontes = ", ".join([f"Doc {c['id']}" for c in ctx])
            clean = f"{clean} (Fontes: {fontes})"

    return {
        "answer": clean,
        "sources": [c["id"] for c in ctx],
        "meta": [c.get("meta", {}) for c in ctx],
        "debug": {"prompt": prompt[:1000]},
    }
