from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, MODEL_CACHE_DIR


def _has_cached_model() -> bool:
    return any(MODEL_CACHE_DIR.glob("**/modules.json"))


def create_embedding_model() -> HuggingFaceEmbeddings:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _has_cached_model():
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                cache_folder=str(MODEL_CACHE_DIR),
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            pass

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=str(MODEL_CACHE_DIR),
        )
    except Exception as error:
        raise RuntimeError(
            "無法載入 HuggingFace embedding model。請確認網路可連到 HuggingFace，"
            "或將已下載好的 .hf-cache/ 一起放在專案根目錄後再重試。"
        ) from error
