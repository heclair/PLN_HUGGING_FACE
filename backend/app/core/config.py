import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def _clean(v: str) -> str:
    return (v or "").strip().strip('"').strip("'")

class Settings:
    HF_TOKEN: str = _clean(os.getenv("HF_TOKEN", ""))
    HF_MODEL: str = _clean(os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta"))
    EMBED_MODEL: str = _clean(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # 🔽 persistência e seed
    INDEX_DIR: str = _clean(os.getenv("INDEX_DIR", "data"))
    PERSIST_INDEX: bool = _clean(os.getenv("PERSIST_INDEX", "1")) == "1"
    AUTO_SEED: bool = _clean(os.getenv("AUTO_SEED", "1")) == "1"   # semea se vazio

    # fallback local (se você já tiver isso)
    HF_USE_LOCAL: bool = _clean(os.getenv("HF_USE_LOCAL", "0")) == "1"
    LOCAL_MODEL: str = _clean(os.getenv("LOCAL_MODEL", "google/flan-t5-small"))

settings = Settings()
