from functools import lru_cache
import math
import re
from typing import List, TypedDict

from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    GEMINI_MODEL,
    MAX_RELEVANT_DISTANCE,
    MIN_RELEVANT_CJK_BIGRAM_OVERLAP,
    MIN_RELEVANT_CJK_BIGRAM_RATIO,
    NO_ANSWER_MESSAGE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from app.embeddings import create_embedding_model
from app.gemini_llm import GeminiApiError, generate_with_gemini
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
    use_gemini: bool
    gemini_api_key: str
    gemini_model: str
    use_llm: bool
    llm_model: str
    retrieved: List[RetrievalItem]
    selected: List[RetrievalItem]
    rag_prompt: str
    answer: str
    answer_mode: str
    sources: List[str]
    confidence_label: str
    gemini_status: str
    gemini_error: str
    llm_status: str
    llm_error: str
    no_answer_reason: str
    steps: List[str]


QUESTION_STOP_PHRASES = (
    "是什麼意思",
    "是甚麼意思",
    "什麼是",
    "甚麼是",
    "是什麼",
    "是甚麼",
    "是誰",
    "為什麼",
    "為何",
    "如何",
    "怎麼",
    "怎樣",
    "哪些",
    "哪個",
    "哪一個",
    "多少",
    "請問",
    "幫我",
    "可以",
    "需要",
    "什麼",
    "甚麼",
    "問題",
)


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


def _compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def _compact_query(text: str) -> str:
    cleaned = _compact_text(text)
    for phrase in QUESTION_STOP_PHRASES:
        cleaned = cleaned.replace(phrase, "")

    return "".join(char for char in cleaned if char not in {"是", "的", "了", "嗎", "呢", "啊"})


def _ascii_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower()))


def _cjk_sequences(text: str) -> list[str]:
    return [match for match in re.findall(r"[\u4e00-\u9fff]+", text) if len(match) >= 2]


def _cjk_bigrams(text: str) -> set[str]:
    bigrams = set()
    for sequence in _cjk_sequences(text):
        if len(sequence) == 2:
            bigrams.add(sequence)
        else:
            bigrams.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return bigrams


def _has_query_anchor(question: str, content: str) -> bool:
    compact_query = _compact_query(question)
    compact_content = _compact_text(content)

    ascii_terms = _ascii_terms(question)
    if ascii_terms and any(term in compact_content for term in ascii_terms):
        return True

    cjk_terms = _cjk_sequences(compact_query)
    if any(2 <= len(term) <= 8 and term in compact_content for term in cjk_terms):
        return True

    query_bigrams = _cjk_bigrams(compact_query)
    if not query_bigrams:
        return False

    content_bigrams = _cjk_bigrams(compact_content)
    overlap_count = len(query_bigrams.intersection(content_bigrams))
    required_overlap = min(
        len(query_bigrams),
        max(
            MIN_RELEVANT_CJK_BIGRAM_OVERLAP,
            math.ceil(len(query_bigrams) * MIN_RELEVANT_CJK_BIGRAM_RATIO),
        ),
    )
    return overlap_count >= required_overlap


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
    top_distance = retrieved[0]["distance"] if retrieved else None
    confidence_label = "Low"
    distance_candidates = [
        item for item in retrieved if item["distance"] <= MAX_RELEVANT_DISTANCE
    ]

    if top_distance is None:
        selected = []
        no_answer_reason = "Chroma 沒有取回任何文件片段。"
    elif top_distance > MAX_RELEVANT_DISTANCE:
        selected = []
        no_answer_reason = (
            f"最接近的文件距離為 {top_distance}，高於門檻 {MAX_RELEVANT_DISTANCE}。"
        )
    else:
        selected = [
            item
            for item in distance_candidates
            if _has_query_anchor(state["question"], item["content"])
        ][:3]
        if selected:
            no_answer_reason = ""
        else:
            no_answer_reason = "取回片段未命中問題中的關鍵詞或命名實體。"

    top_confidence = selected[0]["confidence"] if selected else 0
    if top_confidence >= 0.7 and not no_answer_reason:
        confidence_label = "High"
    elif top_confidence >= 0.45 and not no_answer_reason:
        confidence_label = "Medium"

    return {
        **state,
        "selected": selected,
        "confidence_label": confidence_label,
        "no_answer_reason": no_answer_reason,
        "steps": [*state.get("steps", []), "rank_and_select_context"],
    }


def _format_sources(selected: List[RetrievalItem]) -> List[str]:
    return [f"{item['source']}#chunk-{item['chunk_id']}" for item in selected]


def _append_sources(answer: str, sources: List[str]) -> str:
    if sources and not any(source in answer for source in sources):
        return f"{answer.rstrip()}\n\n引用來源：{', '.join(sources)}"
    return answer


def _join_status(*messages: str) -> str:
    return "\n".join(message for message in messages if message)


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
            + "\n\n請用繁體中文回答。若文件沒有足夠資訊，"
            f"請只回覆「{NO_ANSWER_MESSAGE}」。不要使用文件外的知識。"
            "回答最後列出引用來源。"
        )

    return {
        **state,
        "rag_prompt": rag_prompt,
        "sources": _format_sources(selected),
        "steps": [*state.get("steps", []), "build_rag_prompt"],
    }


def generate_answer(state: GraphState) -> GraphState:
    selected = state.get("selected", [])
    step_name = "generate_answer"

    if not selected:
        return {
            **state,
            "answer": NO_ANSWER_MESSAGE,
            "answer_mode": "No Context",
            "steps": [*state.get("steps", []), step_name],
        }

    sources = state.get("sources", [])
    gemini_status = ""
    gemini_error = ""

    if state.get("use_gemini", True):
        gemini_model = state.get("gemini_model") or GEMINI_MODEL
        try:
            answer = generate_with_gemini(
                state.get("rag_prompt", ""),
                model=gemini_model,
                api_key=state.get("gemini_api_key") or None,
            )
            return {
                **state,
                "answer": _append_sources(answer, sources),
                "answer_mode": "Gemini RAG",
                "gemini_status": f"Gemini API 已成功使用模型 {gemini_model} 生成回答。",
                "steps": [*state.get("steps", []), step_name],
            }
        except GeminiApiError as exc:
            gemini_error = str(exc)
            if exc.resource_exhausted:
                gemini_status = "Gemini API 額度已用完（429 RESOURCE_EXHAUSTED），已切換到本機 LLM 備援。"
            else:
                gemini_status = f"Gemini API 無法使用，已切換到本機 LLM 備援。原因：{exc}"
    else:
        gemini_status = "Gemini API 已停用，改用下一層備援。"

    if not state.get("use_llm", True):
        return {
            **state,
            "answer": _build_extractive_answer(selected),
            "answer_mode": "Extractive RAG",
            "gemini_status": gemini_status,
            "gemini_error": gemini_error,
            "llm_status": "本機 LLM 備援已停用。",
            "steps": [*state.get("steps", []), step_name],
        }

    llm_model = state.get("llm_model") or OLLAMA_MODEL
    status = check_local_llm(model=llm_model, base_url=OLLAMA_BASE_URL)

    if not status.available:
        return {
            **state,
            "answer": _build_extractive_answer(selected),
            "answer_mode": "Extractive RAG Fallback",
            "gemini_status": gemini_status,
            "gemini_error": gemini_error,
            "llm_status": _join_status(gemini_status, f"本機 LLM 不可用：{status.message}"),
            "llm_error": _join_status(gemini_error, status.message),
            "steps": [*state.get("steps", []), step_name],
        }

    try:
        answer = generate_with_ollama(
            state.get("rag_prompt", ""),
            model=llm_model,
            base_url=OLLAMA_BASE_URL,
        )
    except Exception as exc:
        return {
            **state,
            "answer": _build_extractive_answer(selected),
            "answer_mode": "Extractive RAG Fallback",
            "gemini_status": gemini_status,
            "gemini_error": gemini_error,
            "llm_status": _join_status(gemini_status, f"本機 LLM 生成失敗：{exc}"),
            "llm_error": _join_status(gemini_error, str(exc)),
            "steps": [*state.get("steps", []), step_name],
        }

    return {
        **state,
        "answer": _append_sources(answer, sources),
        "answer_mode": "Local LLM RAG",
        "gemini_status": gemini_status,
        "gemini_error": gemini_error,
        "llm_status": _join_status(gemini_status, status.message),
        "steps": [*state.get("steps", []), step_name],
    }


def build_retrieval_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_from_chroma", retrieve_from_chroma)
    workflow.add_node("rank_and_select_context", rank_and_select_context)
    workflow.add_node("build_rag_prompt", build_rag_prompt)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("retrieve_from_chroma")
    workflow.add_edge("retrieve_from_chroma", "rank_and_select_context")
    workflow.add_edge("rank_and_select_context", "build_rag_prompt")
    workflow.add_edge("build_rag_prompt", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_retrieval_graph():
    return build_retrieval_graph()


def run_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    use_gemini: bool = True,
    gemini_api_key: str = "",
    gemini_model: str = GEMINI_MODEL,
    use_llm: bool = True,
    llm_model: str = OLLAMA_MODEL,
) -> GraphState:
    graph = get_retrieval_graph()
    return graph.invoke(
        {
            "question": question,
            "top_k": top_k,
            "use_gemini": use_gemini,
            "gemini_api_key": gemini_api_key,
            "gemini_model": gemini_model,
            "use_llm": use_llm,
            "llm_model": llm_model,
            "steps": [],
        }
    )
