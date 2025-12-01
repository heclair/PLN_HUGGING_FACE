# RAG + Hugging Face (Aula 3)

Implementação completa de um **chatbot com RAG** (Retrieval-Augmented Generation) usando **FastAPI**, **FAISS / embeddings**, **Hugging Face (remoto ou local)** e um **frontend em Streamlit** com UI de ingestão e chat.

## ✨ O que o projeto oferece

- **API FastAPI** com:
  - Ingestão de textos e arquivos `.txt` (com **chunking opcional**)
  - Indexação vetorial (FAISS/in-memory) via **Sentence-Transformers**
  - **RAG**: recuperação dos k documentos mais similares + chamada ao LLM
  - **Chat** com histórico + **prompt engineering** básico
  - Rotas de diagnóstico: health, config e teste do LLM
- **Fallback local** do LLM (Transformers) caso a Inference API não esteja disponível
- **Frontend Streamlit**:
  - Coluna esquerda: **ingestão** com cards dos trechos recém-ingeridos
  - Coluna direita: **chat** com bolhas, **chips de fontes**, presets e “tons” (conciso/explicativo/infantil)
- Pronto para separar **backend** e **frontend** (cada um com seu `.env` e `requirements.txt`)

---

## 🌐 Por que a Inference API do Hugging Face não funcionou

Durante os testes, as chamadas para:
```
POST https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct
```
retornaram **404 Not Found**, mesmo com:
- Token válido (`whoami-v2` OK)
- Model ID correto
- License aceita / acesso ao repositório

**Causas típicas**:

1) **Model ID vs disponibilidade na Serverless API**  
   Alguns repositórios (sobretudo **gated**, como Llama 3.1) não estão habilitados na **Serverless Inference API** pública; o metadata responde, mas a rota de inferência retorna **404**.

2) **Licença “gated”**  
   Aceitar a licença dá acesso ao **repo**, mas não implica acesso à **serverless**. Para inferência gerenciada, use **Inference Endpoints** (pago) ou rode localmente.

3) **Capacidade/região**  
   Em certos momentos a serverless pode não servir modelos maiores, devolvendo 404.

4) **Erros de token/headers** nas tentativas manuais (menos provável no backend).

**Contorno**:  
- Usar o **fallback local** (`HF_USE_LOCAL=1`) — já implementado.  
- Testar com modelos **públicos** da serverless (ex.: `HuggingFaceH4/zephyr-7b-beta`, `google/flan-t5-small`).  
- Para Llama 3.1, preferir **Inference Endpoints** ou **self-host** (Transformers/TGI).

---

## 🧱 Stack

- **Backend**: Python, FastAPI, Uvicorn, FAISS (ou in-memory), Sentence-Transformers
- **LLM (remoto)**: Hugging Face **Inference API**
- **LLM (local)**: `transformers` + modelo leve (ex.: `google/flan-t5-small`)
- **Frontend**: Streamlit

---

## 📁 Estrutura (resumo)

```
.
├─ app/
│  ├─ main.py
│  ├─ routes/
│  │  ├─ health.py
│  │  ├─ ingest.py
│  │  ├─ query.py
│  │  └─ chat.py
│  ├─ services/
│  │  ├─ bootstrap.py
│  │  ├─ embeddings.py
│  │  ├─ index.py
│  │  └─ rag.py
│  ├─ core/
│  │  ├─ config.py
│  │  └─ llm.py
│  └─ models/
│     └─ schemas.py
├─ frontend/
│  └─ app.py
├─ data/                 # (opcional) índice persistido
├─ .env                  # backend
├─ frontend/.env         # frontend
└─ requirements.txt
```

---

## ⚙️ Configuração (Backend)

### 1) Pré-requisitos
- Python **3.10+** (recomendado 3.10–3.12)
- Windows: PowerShell ou CMD; Linux/macOS: bash/zsh

### 2) Ambiente virtual e dependências
```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) `.env` (backend) — exemplo
> **Não** comitar o `.env`.

```env
# HF remoto (opcional)
HF_TOKEN=hf_xxx_sua_chave
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Embeddings (PT): robusto a minúsculas/acentos
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Fallback local (recomendado manter habilitado)
HF_USE_LOCAL=1
LOCAL_MODEL=google/flan-t5-small

# Persistência/seed do índice
INDEX_DIR=data
PERSIST_INDEX=1
AUTO_SEED=1
```

> Se quiser **apenas local**, é suficiente `HF_USE_LOCAL=1` e `LOCAL_MODEL=google/flan-t5-small` (ou outro leve).

### 4) Subir o backend
```bash
uvicorn app.main:app --reload --port 8000
```
- Swagger: http://127.0.0.1:8000/docs  
- Health: `GET /health`

---

## 🖥️ Configuração (Frontend)

### 1) Ambiente e dependências
No diretório `frontend/`:

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install streamlit python-dotenv requests
```

```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install streamlit python-dotenv requests
```

### 2) `frontend/.env` (exemplo)
```env
BACKEND_URL=http://127.0.0.1:8000
CHAT_DEFAULT_TOP_K=3
CHAT_DEFAULT_TEMPERATURE=0.2
CHAT_DEFAULT_MAX_NEW_TOKENS=120
```

### 3) Rodar o front
```bash
streamlit run frontend/app.py
```

---

## 🔌 API — Visão geral das rotas

### Saúde & Debug
- `GET /health` → `{"status":"ok", "docs": <int>, "index_built": <bool>}`
- `GET /debug/config` → mostra o que a API carregou do `.env` (oculta o token)
- `GET /debug/hf` → **teste do LLM atual** (respeita fallback/local)
- `GET /debug/hf-remote` → força **Inference API** (sem fallback) — útil para diagnosticar

### Ingestão
- `POST /ingest/sample`  
  Ingesta 3 textos de exemplo.
- `POST /ingest/texts`  
  **Body**:
  ```json
  { "texts": ["texto 1", "texto 2"], "chunk": true }
  ```
- `POST /ingest/file`  
  **Multipart**: `file=<.txt>`, `chunk=true|false`

### RAG “simples”
- `POST /query`  
  **Body**:
  ```json
  {
    "question": "O que é RAG?",
    "top_k": 3,
    "temperature": 0.2,
    "max_new_tokens": 128
  }
  ```

### Chat (com histórico)
- `POST /chat`  
  **Body**:
  ```json
  {
    "session_id": "demo",
    "message": "Explique RAG em 1 frase.",
    "top_k": 3,
    "temperature": 0.2,
    "max_new_tokens": 120,
    "system_prompt": "opcional (muda o tom da resposta)"
  }
  ```
  **Resposta**: `answer`, `sources` (ids dos docs usados), `meta` (metadados), `debug.prompt`.

> No frontend, os **chips** exibem as fontes: `Doc 0`, `Doc 1`, ...

---

## 🧠 Embeddings e relevância (PT-BR)

- Embeddings definidos em `.env` via `EMBED_MODEL`.  
  Recomendado:  
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- O serviço normaliza os textos (casefold + acentos), reduzindo sensibilidade a **minúsculas/maiúsculas** (ex.: “brasil” vs “Brasil”).
- O limiar de similaridade (`MIN_SIM`, em `rag.py`) foi ajustado para PT.  
  Se notar respostas “não sei” com textos parecidos, **reduza um pouco** esse limiar.

---

## ▶️ Fluxo de uso

1. Suba o backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. Suba o frontend:
   ```bash
   streamlit run frontend/app.py
   ```
3. Ingerir exemplos/textos e perguntar no chat.  
   As respostas citam as **fontes** (Doc X, Doc Y).

---
