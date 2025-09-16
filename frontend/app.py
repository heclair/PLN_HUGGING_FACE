import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# ===================== Config =====================
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

DEF_TOP_K   = int(os.getenv("CHAT_DEFAULT_TOP_K", 3))
DEF_TEMP    = float(os.getenv("CHAT_DEFAULT_TEMPERATURE", 0.2))
DEF_MAXTOK  = int(os.getenv("CHAT_DEFAULT_MAX_NEW_TOKENS", 120))

st.set_page_config(page_title="RAG + Hugging Face", page_icon="🧠", layout="wide")

# ===================== CSS (look & feel) =====================
st.markdown("""
<style>
:root {
  --surface: #ffffff;
  --ink: #111827;
  --muted: #6B7280;
  --line: #E5E7EB;
  --brand: #4F46E5;
  --chip-bg: #EEF2FF;
}
.block-container {padding-top: 1rem; padding-bottom: 2rem;}

/* Header */
.app-header {display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;}
.app-title {font-weight:800; font-size:22px; color:var(--ink);}
.app-sub {font-size:12px; color:var(--muted);}

/* Columns separator hint */
.hrspace {margin: 8px 0 16px; border:none; border-top:1px solid var(--line);}

/* Cards de textos ingeridos */
.grid {display:grid; grid-template-columns: repeat(1, 1fr); gap:10px;}
@media (min-width: 900px) {.grid {grid-template-columns: repeat(2, 1fr);} }
@media (min-width: 1200px){.grid {grid-template-columns: repeat(3, 1fr);} }
.doccard {
  border:1px solid var(--line); border-radius:12px; background:#FBFBFE; padding:12px 14px;
}
.doccard small {color:var(--muted); font-size:11px;}
.doctext {font-size:13px; color:#1F2937; margin-top:6px; white-space:pre-wrap;}

/* Chips de fonte */
.chips {display:flex; flex-wrap:wrap; gap:6px; margin-top:.25rem}
.chip {background:var(--chip-bg); color:#1F2937; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--line);}

/* Chat bolhas */
.stChatMessage {border-radius:16px; padding:10px 12px;}

/* Presets */
.preset-row {display:flex; gap:8px; flex-wrap:wrap; margin:4px 0 8px;}
.preset {border:1px solid var(--line); color:#1F2937; background:var(--surface);
  border-radius:999px; padding:4px 10px; font-size:12px; cursor:pointer;}
.preset:hover {border-color:#C7D2FE;}

/* Subtle card */
.card {border:1px solid var(--line); border-radius:12px; background:var(--surface); padding:12px 14px;}
</style>
""", unsafe_allow_html=True)

# ===================== HTTP helpers =====================
def _post_json(url: str, payload: dict, timeout=60):
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    dt = int((time.perf_counter() - t0) * 1000.0)
    r.raise_for_status()
    return r.json(), dt

def _post_multipart(url: str, files, data: dict, timeout=120):
    t0 = time.perf_counter()
    r = requests.post(url, files=files, data=data, timeout=timeout)
    dt = int((time.perf_counter() - t0) * 1000.0)
    r.raise_for_status()
    return r.json(), dt

# ===================== utils =====================
def _snippet(txt: str, limit=260):
    txt = " ".join((txt or "").split())
    return (txt[:limit] + "…") if len(txt) > limit else txt

def chip_list(ids):
    if not ids: return ""
    return "".join([f'<span class="chip">Doc {i}</span>' for i in ids])

def remember_snippets(items, source):
    """Armazena no estado do front trechos dos textos ingeridos (apenas visual)."""
    if "ingested_snippets" not in st.session_state:
        st.session_state.ingested_snippets = []
    for t in items:
        t = (t or "").strip()
        if not t: 
            continue
        st.session_state.ingested_snippets.append({
            "source": source,
            "text": _snippet(t, 260)
        })

# ===================== Header =====================
st.markdown(
    f"""
    <div class="app-header">
      <div>
        <div class="app-title">🧠 RAG + Hugging Face</div>
        <div class="app-sub">Backend: {BACKEND_URL}</div>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# ===================== layout principal: ingestão | chat =====================
left, right = st.columns([1.1, 1.9], gap="large")

# --------------------- Ingestão ---------------------
with left:
    st.markdown("### 📥 Ingestão")

    with st.container():
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Ingerir exemplos", use_container_width=True):
                try:
                    with st.spinner("Indexando amostras…"):
                        _res, _ = _post_json(f"{BACKEND_URL}/ingest/sample", {})
                    # Mostra localmente:
                    sample_texts = [
                        "RAG combina recuperação de informação com geração de texto, melhorando precisão.",
                        "Hugging Face Hub oferece modelos, datasets e spaces para IA.",
                        "Prompt Engineering é a prática de desenhar prompts para melhorar a resposta de LLMs.",
                    ]
                    remember_snippets(sample_texts, "amostras")
                    st.toast("Amostras ingeridas!", icon="✅")
                except Exception as e:
                    st.toast(f"Falha: {e}", icon="❌")
        with col2:
            chunk = st.checkbox("Fazer chunk", value=True, help="Divide o texto em pedaços antes de indexar")

    st.markdown("#### Texto")
    txt = st.text_area(
        "Cole um ou mais textos (um por linha):",
        height=150,
        placeholder="Ex.: \nA capital do Brasil é Brasília\nO melhor café do mundo é do Brasil"
    )
    if st.button("Ingerir textos", use_container_width=True):
        lines = [t.strip() for t in txt.splitlines() if t.strip()]
        if not lines:
            st.warning("Adicione pelo menos um texto.")
        else:
            try:
                payload = {"texts": lines, "chunk": bool(chunk)}
                with st.spinner("Indexando textos…"):
                    res, _ = _post_json(f"{BACKEND_URL}/ingest/texts", payload)
                remember_snippets(lines, "textarea")
                st.success(f"Ingeridos: {res.get('ingested','?')} • Total: {res.get('total_docs','?')}")
            except Exception as e:
                st.error(f"Falha ao ingerir: {e}")

    st.markdown("#### Arquivo .txt")
    up = st.file_uploader("Selecione um .txt", type=["txt"])
    if up is not None and st.button("Ingerir arquivo", use_container_width=True):
        try:
            content_bytes = up.getvalue()
            preview = content_bytes.decode("utf-8", errors="ignore")
            lines = [l.strip() for l in preview.splitlines() if l.strip()]
            snippets = lines[:4] if lines else [preview[:280]]

            files = {"file": (up.name, content_bytes, "text/plain")}
            data = {"chunk": str(chunk).lower()}
            with st.spinner("Indexando arquivo…"):
                _res, _ = _post_multipart(f"{BACKEND_URL}/ingest/file", files, data)
            remember_snippets(snippets, f"arquivo:{up.name}")
            st.success(f"Ingerido: {up.name}")
        except Exception as e:
            st.error(f"Falha ao ingerir arquivo: {e}")

    st.markdown("#### Textos recém-ingeridos (sessão)")
    cA, cB = st.columns([1,1])
    with cA:
        if st.button("Limpar lista (frontend)", use_container_width=True):
            st.session_state.ingested_snippets = []
    with cB:
        st.caption("Somente o que você enviou nesta sessão (visual).")

    grid = st.session_state.get("ingested_snippets", [])
    if not grid:
        st.info("Nada por aqui ainda. Ingerir textos para aparecerem os cards.")
    else:
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        for item in grid:
            st.markdown(
                f'<div class="doccard"><small>{item.get("source","")}</small>'
                f'<div class="doctext">{item.get("text","")}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------- Chat ---------------------
with right:
    st.markdown("### 💬 Chat")

    # Estado inicial do chat
    if "messages" not in st.session_state:
        st.session_state.messages = []   # [{role:'user'|'assistant', 'content': str, 'sources':[ids]}]

    # Parâmetros e "tons" de resposta
    with st.expander("Parâmetros", expanded=False):
        session_id      = st.text_input("session_id", value="demo")
        top_k           = st.slider("top_k (docs)", 1, 10, DEF_TOP_K)
        temperature     = st.slider("temperature", 0.0, 1.5, DEF_TEMP, 0.1)
        max_new_tokens  = st.slider("max_new_tokens", 16, 512, DEF_MAXTOK, 16)

        tone = st.radio(
            "Tom da resposta",
            options=["Conciso (1 frase)", "Explicativo (3 frases)", "Infantil (5 anos)"],
            index=0,
            horizontal=True
        )
        tone_map = {
            "Conciso (1 frase)": "Você é um assistente técnico e objetivo. Responda em 1 frase com fontes.",
            "Explicativo (3 frases)": "Você é um assistente didático. Responda em até 3 frases com fontes.",
            "Infantil (5 anos)": "Explique como se eu tivesse 5 anos, simples e gentil. Termine com as fontes."
        }
        system_prompt = tone_map[tone]

        colx, coly = st.columns([1,1])
        with colx:
            if st.button("Limpar conversa", use_container_width=True):
                st.session_state.messages = []
                st.toast("Histórico do chat limpo.", icon="🧹")
        with coly:
            st.caption("Dica: pergunte algo que está nos textos ingeridos 😉")

    # Presets rápidos
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Sugestões rápidas")
    preset_row = st.container()
    with preset_row:
        cols = st.columns([1,1,1,1])
        presets = [
            "Explique RAG em 1 frase.",
            "Resuma o Hugging Face Hub em 2 frases.",
            "O que é Prompt Engineering?",
            "Qual a capital do Brasil?"
        ]
        for i, p in enumerate(presets):
            if cols[i % 4].button(p, key=f"preset_{i}"):
                st.session_state.messages.append({"role": "user", "content": p})
                # Render imediato
                with st.chat_message("user"): st.markdown(p)
                # Dispara requisição
                body = {
                    "session_id": session_id,
                    "message": p,
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "system_prompt": system_prompt,
                }
                try:
                    with st.spinner("Gerando resposta…"):
                        data, dt = _post_json(f"{BACKEND_URL}/chat", body, timeout=120)
                    ans   = data.get("answer", "")
                    srcs  = data.get("sources", [])
                    with st.chat_message("assistant"):
                        st.markdown(ans or "(sem resposta)")
                        if srcs:
                            st.markdown(f'<div class="chips">{chip_list(srcs)}</div>', unsafe_allow_html=True)
                        st.caption(f"{dt} ms")
                    st.session_state.messages.append({"role":"assistant","content":ans, "sources":srcs})
                except requests.HTTPError as e:
                    with st.chat_message("assistant"):
                        st.error(f"Erro {e.response.status_code}: {e.response.text}")
                    st.session_state.messages.append({"role":"assistant","content":f"[Erro] {e}"})
                except Exception as e:
                    with st.chat_message("assistant"):
                        st.error(f"Falha na chamada /chat: {e}")
                    st.session_state.messages.append({"role":"assistant","content":f"[Falha] {e}"})
    st.markdown('</div>', unsafe_allow_html=True)

    # Histórico
    for m in st.session_state.messages:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])
            if m.get("sources"):
                st.markdown(f'<div class="chips">{chip_list(m["sources"])}</div>', unsafe_allow_html=True)

    # Input
    user_msg = st.chat_input("Escreva sua pergunta aqui…")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"): st.markdown(user_msg)

        body = {
            "session_id": session_id,
            "message": user_msg,
            "top_k": int(top_k),
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            "system_prompt": system_prompt,
        }
        try:
            with st.spinner("Gerando resposta…"):
                data, dt = _post_json(f"{BACKEND_URL}/chat", body, timeout=120)
            ans   = data.get("answer", "")
            srcs  = data.get("sources", [])
            with st.chat_message("assistant"):
                st.markdown(ans or "(sem resposta)")
                if srcs:
                    st.markdown(f'<div class="chips">{chip_list(srcs)}</div>', unsafe_allow_html=True)
                st.caption(f"{dt} ms")
            st.session_state.messages.append({"role":"assistant","content":ans, "sources":srcs})
        except requests.HTTPError as e:
            with st.chat_message("assistant"):
                st.error(f"Erro {e.response.status_code}: {e.response.text}")
            st.session_state.messages.append({"role":"assistant","content":f"[Erro] {e}"})
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Falha na chamada /chat: {e}")
            st.session_state.messages.append({"role":"assistant","content":f"[Falha] {e}"})
