from functools import lru_cache
from typing import List, TypedDict

from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from app.embeddings import create_embedding_model
from app.local_llm import check_local_llm, generate_with_ollama


class RetrievalItem(TypedDict):
    rank: int
    source: str
    chunk_id: int
    distance: float
    confidence: float
    preview: str
    content: str


class GraphState(TypedDict, total=False):
    question: str
    top_k: int
    use_llm: bool
    llm_model: str
    retrieved: List[RetrievalItem]
    selected: List[RetrievalItem]
    rag_prompt: str
    answer: str
    answer_mode: str
    sources: List[str]
    confidence_label: str
    llm_status: str
    llm_error: str
    steps: List[str]


@lru_cache(maxsize=1)
def get_embedding_model():
    return create_embedding_model()


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
    )


def clear_runtime_caches() -> None:
    get_vector_db.cache_clear()
    get_embedding_model.cache_clear()
    get_retrieval_graph.cache_clear()


def _normalize_text(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _distance_to_match_score(distance: float, best: float, worst: float) -> float:
    if worst <= best:
        return 1.0

    relative_score = 1 - ((distance - best) / (worst - best))
    return round(0.45 + (relative_score * 0.5), 4)


def _source_name(metadata: dict) -> str:
    return metadata.get("source") or metadata.get("filename") or "unknown"


def retrieve_from_chroma(state: GraphState) -> GraphState:
    question = state["question"].strip()
    top_k = int(state.get("top_k") or DEFAULT_TOP_K)

    if not question:
        return {
            **state,
            "retrieved": [],
            "steps": [*state.get("steps", []), "retrieve_from_chroma"],
        }

    vector_db = get_vector_db()
    docs_with_scores = vector_db.similarity_search_with_score(question, k=top_k)
    distances = [float(distance) for _, distance in docs_with_scores]
    best_distance = min(distances) if distances else 0.0
    worst_distance = max(distances) if distances else 0.0

    retrieved: List[RetrievalItem] = []
    for rank, (doc, distance) in enumerate(docs_with_scores, start=1):
        distance_value = float(distance)
        metadata = doc.metadata or {}
        retrieved.append(
            {
                "rank": rank,
                "source": _source_name(metadata),
                "chunk_id": int(metadata.get("chunk_id") or 0),
                "distance": round(distance_value, 4),
                "confidence": _distance_to_match_score(
                    distance_value,
                    best_distance,
                    worst_distance,
                ),
                "preview": _normalize_text(doc.page_content),
                "content": doc.page_content,
            }
        )

    return {
        **state,
        "retrieved": retrieved,
        "steps": [*state.get("steps", []), "retrieve_from_chroma"],
    }


def rank_and_select_context(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", [])
    selected = retrieved[: min(3, len(retrieved))]

    top_confidence = selected[0]["confidence"] if selected else 0
    if top_confidence >= 0.7:
        confidence_label = "High"
    elif top_confidence >= 0.45:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return {
        **state,
        "selected": selected,
        "confidence_label": confidence_label,
        "steps": [*state.get("steps", []), "rank_and_select_context"],
    }


def _format_sources(selected: List[RetrievalItem]) -> List[str]:
    return [f"{item['source']}#chunk-{item['chunk_id']}" for item in selected]


def _build_extractive_answer(selected: List[RetrievalItem]) -> str:
    evidence_lines = []

    for item in selected:
        label = f"{item['source']}#chunk-{item['chunk_id']}"
        evidence_lines.append(f"- {label}: {item['preview']}")

    return (
        "根據 Chroma 取回的文件片段，這個問題可由下列內部知識回答：\n\n"
        + "\n".join(evidence_lines)
        + "\n\n這是一個離線 extractive RAG demo：LangGraph 負責流程編排，"
        "LangChain 串接 embedding/vector store，Chroma 提供本地語意檢索與來源追蹤。"
    )


def build_rag_prompt(state: GraphState) -> GraphState:
    selected = state.get("selected", [])

    if not selected:
        rag_prompt = ""
    else:
        context_blocks = []
        for item in selected:
            context_blocks.append(
                f"[{item['source']}#chunk-{item['chunk_id']}]\n{item['content']}"
            )
        rag_prompt = (
            f"使用者問題：{state['question']}\n\n"
            "可用內部文件內容：\n"
            + "\n\n---\n\n".join(context_blocks)
            + "\n\n請用繁體中文回答。若文件沒有足夠資訊，請明確說明不足，"
            "不要使用文件外的知識。回答最後列出引用來源。"
        )

    return {
        **state,
        "rag_prompt": rag_prompt,
        "sources": _format_sources(selected),
        "steps": [*state.get("steps", []), "build_rag_prompt"],
    }


def generate_with_local_llm(state: GraphState) -> GraphState:
    selected = state.get("selected", [])

    if not selected:
        return {
            **state,
            "answer": (
                "目前 Chroma collection 中沒有找到足夠內容。請先執行 "
                "`python -m app.ingest` 重建向量資料庫，或把更多 .txt 文件放進 data/。"
            ),
            "answer_mode": "No Context",
            "steps": [*state.get("steps", []), "generate_with_local_llm"],
        }

    if not state.get("use_llm", False):
        return {
            **state,
            "answer": _build_extractive_answer(selected),
            "answer_mode": "Extractive RAG",
            "llm_status": "Local LLM disabled",
            "steps": [*state.get("steps", []), "generate_with_local_llm"],
        }

    llm_model = state.get("llm_model") or OLLAMA_MODEL
    status = check_local_llm(model=llm_model, base_url=OLLAMA_BASE_URL)

    if not status.available:
        fallback_answer = _build_extractive_answer(selected)
        return {
            **state,
            "answer": (
                f"Local LLM 尚未就緒，已自動使用 extractive RAG fallback。\n\n"
                f"狀態：{status.message}\n\n"
                f"{fallback_answer}"
            ),
            "answer_mode": "Extractive RAG Fallback",
            "llm_status": status.message,
            "llm_error": status.message,
            "steps": [*state.get("steps", []), "generate_with_local_llm"],
        }

    try:
        answer = generate_with_ollama(
            state.get("rag_prompt", ""),
            model=llm_model,
            base_url=OLLAMA_BASE_URL,
        )
        sources = state.get("sources", [])
        if sources and not any(source in answer for source in sources):
            answer = f"{answer.rstrip()}\n\n引用來源：{', '.join(sources)}"
    except Exception as exc:
        fallback_answer = _build_extractive_answer(selected)
        return {
            **state,
            "answer": (
                f"Local LLM 生成時發生錯誤，已自動使用 extractive RAG fallback。\n\n"
                f"錯誤：{exc}\n\n"
                f"{fallback_answer}"
            ),
            "answer_mode": "Extractive RAG Fallback",
            "llm_status": status.message,
            "llm_error": str(exc),
            "steps": [*state.get("steps", []), "generate_with_local_llm"],
        }

    return {
        **state,
        "answer": answer,
        "answer_mode": "Local LLM RAG",
        "llm_status": status.message,
        "steps": [*state.get("steps", []), "generate_with_local_llm"],
    }


def build_retrieval_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_from_chroma", retrieve_from_chroma)
    workflow.add_node("rank_and_select_context", rank_and_select_context)
    workflow.add_node("build_rag_prompt", build_rag_prompt)
    workflow.add_node("generate_with_local_llm", generate_with_local_llm)

    workflow.set_entry_point("retrieve_from_chroma")
    workflow.add_edge("retrieve_from_chroma", "rank_and_select_context")
    workflow.add_edge("rank_and_select_context", "build_rag_prompt")
    workflow.add_edge("build_rag_prompt", "generate_with_local_llm")
    workflow.add_edge("generate_with_local_llm", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_retrieval_graph():
    return build_retrieval_graph()


def run_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    use_llm: bool = False,
    llm_model: str = OLLAMA_MODEL,
) -> GraphState:
    graph = get_retrieval_graph()
    return graph.invoke(
        {
            "question": question,
            "top_k": top_k,
            "use_llm": use_llm,
            "llm_model": llm_model,
            "steps": [],
        }
    )
