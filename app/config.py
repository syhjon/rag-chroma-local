from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
MODEL_CACHE_DIR = BASE_DIR / ".hf-cache"

COLLECTION_NAME = "local_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 4
MAX_RELEVANT_DISTANCE = 20.0
MIN_RELEVANT_CJK_BIGRAM_OVERLAP = 2
MIN_RELEVANT_CJK_BIGRAM_RATIO = 0.25
NO_ANSWER_MESSAGE = "目前資料庫無此答案，請問其他問題"

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TIMEOUT_SECONDS = 30
GEMINI_MIN_SECONDS_BETWEEN_CALLS = 8

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
