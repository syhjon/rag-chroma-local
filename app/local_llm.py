from dataclasses import dataclass
import json
from typing import List
from urllib import error, request

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None


@dataclass(frozen=True)
class LocalLLMStatus:
    available: bool
    package_installed: bool
    server_running: bool
    model_available: bool
    model: str
    installed_models: List[str]
    message: str


def _ollama_url(path: str, base_url: str = OLLAMA_BASE_URL) -> str:
    return f"{base_url.rstrip('/')}{path}"


def list_ollama_models(base_url: str = OLLAMA_BASE_URL, timeout: float = 1.5) -> List[str]:
    with request.urlopen(_ollama_url("/api/tags", base_url), timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return sorted(item["name"] for item in payload.get("models", []) if item.get("name"))


def check_local_llm(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> LocalLLMStatus:
    if ChatOllama is None:
        return LocalLLMStatus(
            available=False,
            package_installed=False,
            server_running=False,
            model_available=False,
            model=model,
            installed_models=[],
            message="尚未安裝 Python 套件 langchain-ollama。",
        )

    try:
        installed_models = list_ollama_models(base_url=base_url)
    except (OSError, TimeoutError, error.URLError, error.HTTPError) as exc:
        return LocalLLMStatus(
            available=False,
            package_installed=True,
            server_running=False,
            model_available=False,
            model=model,
            installed_models=[],
            message=f"無法連線到 Ollama 服務：{base_url}。錯誤：{exc}",
        )

    model_available = model in installed_models
    if not model_available:
        return LocalLLMStatus(
            available=False,
            package_installed=True,
            server_running=True,
            model_available=False,
            model=model,
            installed_models=installed_models,
            message=f"Ollama 已啟動，但尚未安裝模型 {model}。",
        )

    return LocalLLMStatus(
        available=True,
        package_installed=True,
        server_running=True,
        model_available=True,
        model=model,
        installed_models=installed_models,
        message=f"本機 LLM 已就緒：{model}",
    )


def generate_with_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    if ChatOllama is None:
        raise RuntimeError("尚未安裝 langchain-ollama。")

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.1,
        num_predict=700,
    )

    response = llm.invoke(
        [
            (
                "system",
                "你是一位精簡、可靠的企業 RAG 助理。請使用繁體中文回答。"
                "只能根據提供的 context 回答，並用 [source#chunk-id] 標示引用來源。",
            ),
            ("human", prompt),
        ]
    )
    return response.content


def rewrite_query_with_ollama(
    question: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    if ChatOllama is None:
        raise RuntimeError("尚未安裝 langchain-ollama。")

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        num_predict=220,
    )

    response = llm.invoke(
        [
            (
                "system",
                "你是 RAG 查詢改寫器，只負責把使用者問題整理成更適合向量檢索的文字。"
                "請使用繁體中文，只輸出 JSON，不要回答問題，不要補充文件外事實。"
                "JSON 欄位必須包含 normalized_question、retrieval_query、keywords。"
                "keywords 必須是 3 到 8 個短詞。",
            ),
            (
                "human",
                "請改寫下列問題，讓 Chroma 向量檢索更容易命中相關文件片段。\n\n"
                f"使用者問題：{question}",
            ),
        ]
    )
    return response.content
