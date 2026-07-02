from functools import lru_cache
import json
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
    QUERY_REWRITE_MAX_KEYWORDS,
)
from app.embeddings import create_embedding_model
from app.gemini_llm import GeminiApiError, generate_with_gemini
from app.local_llm import check_local_llm, generate_with_ollama, rewrite_query_with_ollama

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None


PARTIAL_NO_ANSWER_MESSAGE = "部分子問題在目前資料庫中沒有足夠資訊，因此未延伸回答。"


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
    rewrite_query: bool
    rewritten_question: str
    retrieval_query: str
    retrieval_search_text: str
    retrieval_keywords: List[str]
    answer_question: str
    rewrite_status: str
    rewrite_error: str
    retrieved: List[RetrievalItem]
    selected: List[RetrievalItem]
    rag_prompt: str
    answer: str
    answer_mode: str
    sources: List[str]
    confidence_label: str
    gemini_status: str
    gemini_error: str
    gemini_resource_exhausted: bool
    llm_status: str
    llm_error: str
    no_answer_reason: str
    steps: List[str]


QUESTION_STOP_PHRASES = (
    "是什麼意思",
    "是甚麼意思",
    "叫做什麼",
    "叫做甚麼",
    "什麼叫做",
    "甚麼叫做",
    "什叫做",
    "啥叫做",
    "什麼叫",
    "甚麼叫",
    "什叫",
    "啥叫",
    "何謂",
    "什麼是",
    "甚麼是",
    "什是",
    "啥是",
    "誰是",
    "是指什麼",
    "是指甚麼",
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
    "叫做",
    "稱為",
    "稱作",
    "定義",
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
    get_opencc_converter.cache_clear()


@lru_cache(maxsize=1)
def get_opencc_converter():
    if OpenCC is None:
        raise RuntimeError(
            "缺少 opencc-python-reimplemented，請執行 "
            "python -m pip install -r requirements.txt"
        )
    return OpenCC("s2t")


def _to_traditional_zh(text: str) -> str:
    converter = get_opencc_converter()
    return converter.convert(str(text or ""))


def _normalize_text(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(_to_traditional_zh(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def _clean_rewrite_text(value: object, max_chars: int = 180) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)

    cleaned = " ".join(_to_traditional_zh(str(value or "")).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _dedupe_strings(items: List[str], max_items: int | None = None) -> List[str]:
    results: List[str] = []
    seen = set()

    for item in items:
        cleaned = _clean_rewrite_text(item, max_chars=60)
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        results.append(cleaned)
        if max_items and len(results) >= max_items:
            break

    return results


def _distance_to_match_score(distance: float, best: float, worst: float) -> float:
    if worst <= best:
        return 1.0

    relative_score = 1 - ((distance - best) / (worst - best))
    return round(0.45 + (relative_score * 0.5), 4)


def _source_name(metadata: dict) -> str:
    return metadata.get("source") or metadata.get("filename") or "unknown"


def _compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _to_traditional_zh(text).lower())


def _compact_query(text: str) -> str:
    cleaned = _compact_text(text)
    for phrase in QUESTION_STOP_PHRASES:
        cleaned = cleaned.replace(phrase, "")

    stop_chars = {"是", "的", "了", "嗎", "呢", "啊", "什", "甚", "啥", "誰"}
    return "".join(char for char in cleaned if char not in stop_chars)


def _ascii_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower()))


def _cjk_sequences(text: str, min_length: int = 2) -> list[str]:
    return [
        match
        for match in re.findall(r"[\u4e00-\u9fff]+", text)
        if len(match) >= min_length
    ]


def _cjk_bigrams(text: str) -> set[str]:
    bigrams = set()
    for sequence in _cjk_sequences(text):
        if len(sequence) == 2:
            bigrams.add(sequence)
        else:
            bigrams.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return bigrams


def _cjk_ngrams(sequence: str, size: int) -> list[str]:
    if len(sequence) < size:
        return []
    return [sequence[index : index + size] for index in range(len(sequence) - size + 1)]


def _cjk_anchor_terms(text: str) -> List[str]:
    compact_query = _compact_query(text)
    terms: List[str] = []

    for sequence in _cjk_sequences(compact_query):
        if 2 <= len(sequence) <= 8:
            terms.append(sequence)
            continue

        for size in (4, 3):
            terms.extend(_cjk_ngrams(sequence, size))

    if len(compact_query) == 2:
        terms.append(compact_query)

    return _dedupe_strings(terms, max_items=QUERY_REWRITE_MAX_KEYWORDS * 4)


def _fallback_query_keywords(question: str) -> List[str]:
    keywords = sorted(_ascii_terms(question))
    keywords.extend(_cjk_anchor_terms(question))

    return _dedupe_strings(keywords, max_items=QUERY_REWRITE_MAX_KEYWORDS)


def _extract_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("本機 LLM 沒有回傳 JSON。")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("本機 LLM 回傳的 JSON 不是物件。")

    return payload


def _parse_keywords(value: object) -> List[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[,，、;；\n]+", str(value or ""))

    return _dedupe_strings(raw_items, max_items=QUERY_REWRITE_MAX_KEYWORDS)


def _parse_query_rewrite(raw_text: str, question: str) -> dict:
    payload = _extract_json_payload(raw_text)

    normalized_question = _clean_rewrite_text(
        payload.get("normalized_question") or question,
    )
    retrieval_query = _clean_rewrite_text(
        payload.get("retrieval_query") or normalized_question or question,
    )
    keywords = _parse_keywords(payload.get("keywords"))

    if not keywords:
        keywords = _fallback_query_keywords(retrieval_query or normalized_question or question)

    return {
        "rewritten_question": normalized_question or question,
        "retrieval_query": retrieval_query or normalized_question or question,
        "retrieval_keywords": keywords,
    }


def _build_retrieval_search_text(state: GraphState) -> str:
    retrieval_query = state.get("retrieval_query") or state.get("rewritten_question") or state["question"]
    keywords = state.get("retrieval_keywords", [])

    parts = [retrieval_query]
    if keywords:
        parts.append(f"關鍵詞：{'、'.join(keywords)}")

    return "\n".join(_dedupe_strings(parts))


def _extract_noise_terms(question: str) -> List[str]:
    normalized_question = _to_traditional_zh(question)
    terms: List[str] = []

    for pattern in (
        r"我是([^？?。！!，,；;\s]{2,16})",
        r"你是([^？?。！!，,；;\s]{2,16})",
    ):
        terms.extend(re.findall(pattern, normalized_question))

    return _dedupe_strings(terms, max_items=8)


def _strip_irrelevant_question_parts(question: str) -> str:
    normalized_question = _to_traditional_zh(question)
    cleaned = normalized_question

    for pattern in (
        r"我是[^？?。！!，,；;\s]{2,16}[？?。！!，,；;\s]*",
        r"你是[^？?。！!，,；;\s]{2,16}[？?。！!，,；;\s]*",
    ):
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or normalized_question


def _build_answer_question(state: GraphState) -> str:
    original_question = _strip_irrelevant_question_parts(state["question"])
    if len(_compact_text(original_question)) >= 4:
        return original_question

    rewritten_question = state.get("rewritten_question", "").strip()

    if rewritten_question and rewritten_question != state["question"]:
        rewritten_question = _strip_irrelevant_question_parts(rewritten_question)
        if rewritten_question and len(_compact_text(rewritten_question)) >= 4:
            return rewritten_question

    return original_question


def _has_query_anchor(question: str, content: str) -> bool:
    compact_query = _compact_query(question)
    compact_content = _compact_text(content)

    ascii_terms = _ascii_terms(question)
    if ascii_terms and any(term in compact_content for term in ascii_terms):
        return True

    cjk_terms = _cjk_anchor_terms(question)
    if any(term in compact_content for term in cjk_terms):
        return True

    if len(compact_query) == 2 and compact_query in compact_content:
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


def _has_state_relevance_anchor(state: GraphState, content: str) -> bool:
    original_question = state["question"]
    if _has_query_anchor(original_question, content):
        return True

    compact_original = _compact_query(original_question)
    original_has_specific_term = bool(_ascii_terms(original_question)) or any(
        len(term) >= 3 for term in _cjk_sequences(compact_original)
    )
    if original_has_specific_term:
        return False

    auxiliary_question = " ".join(
        [
            state.get("rewritten_question", ""),
            " ".join(state.get("retrieval_keywords", [])),
        ]
    )
    return bool(auxiliary_question.strip()) and _has_query_anchor(auxiliary_question, content)


def rewrite_query_for_retrieval(state: GraphState) -> GraphState:
    question = state["question"].strip()
    step_name = "rewrite_query_for_retrieval"
    fallback = {
        "rewritten_question": question,
        "retrieval_query": question,
        "retrieval_keywords": _fallback_query_keywords(question),
    }

    if not question:
        return {
            **state,
            **fallback,
            "rewrite_status": "問題為空，略過查詢改寫。",
            "steps": [*state.get("steps", []), step_name],
        }

    if not state.get("rewrite_query", True):
        return {
            **state,
            **fallback,
            "rewrite_status": "本機 LLM 查詢改寫已停用，使用原始問題檢索。",
            "steps": [*state.get("steps", []), step_name],
        }

    llm_model = state.get("llm_model") or OLLAMA_MODEL
    status = check_local_llm(model=llm_model, base_url=OLLAMA_BASE_URL)
    if not status.available:
        return {
            **state,
            **fallback,
            "rewrite_status": f"本機 LLM 不可用，使用原始問題檢索：{status.message}",
            "rewrite_error": status.message,
            "steps": [*state.get("steps", []), step_name],
        }

    try:
        raw_rewrite = rewrite_query_with_ollama(
            question,
            model=llm_model,
            base_url=OLLAMA_BASE_URL,
        )
        parsed_rewrite = _parse_query_rewrite(raw_rewrite, question)
    except Exception as exc:
        return {
            **state,
            **fallback,
            "rewrite_status": f"本機 LLM 查詢改寫失敗，使用原始問題檢索：{exc}",
            "rewrite_error": str(exc),
            "steps": [*state.get("steps", []), step_name],
        }

    return {
        **state,
        **parsed_rewrite,
        "rewrite_status": f"本機 LLM 已使用 {llm_model} 完成查詢改寫。",
        "rewrite_error": "",
        "steps": [*state.get("steps", []), step_name],
    }


def retrieve_from_chroma(state: GraphState) -> GraphState:
    search_text = _build_retrieval_search_text(state).strip()
    top_k = int(state.get("top_k") or DEFAULT_TOP_K)

    if not search_text:
        return {
            **state,
            "retrieval_search_text": search_text,
            "retrieved": [],
            "steps": [*state.get("steps", []), "retrieve_from_chroma"],
        }

    vector_db = get_vector_db()
    docs_with_scores = vector_db.similarity_search_with_score(search_text, k=top_k)
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
        "retrieval_search_text": search_text,
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
            if _has_state_relevance_anchor(state, item["content"])
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


def _normalize_partial_no_answer(answer: str) -> str:
    if NO_ANSWER_MESSAGE not in answer:
        return answer

    answer_without_global_no_answer = re.sub(
        rf"\s*{re.escape(NO_ANSWER_MESSAGE)}[。.]?\s*",
        "\n\n",
        answer,
    ).strip()

    has_substantive_answer = len(_compact_text(answer_without_global_no_answer)) >= 8
    if not has_substantive_answer:
        return answer

    if PARTIAL_NO_ANSWER_MESSAGE in answer_without_global_no_answer:
        return answer_without_global_no_answer

    return f"{answer_without_global_no_answer}\n\n{PARTIAL_NO_ANSWER_MESSAGE}"


def _remove_noise_opening(answer: str, question: str) -> str:
    noise_terms = _extract_noise_terms(question)
    if not noise_terms:
        return answer

    paragraphs = [paragraph.strip() for paragraph in answer.split("\n\n")]
    while paragraphs:
        first_paragraph = paragraphs[0]
        first_compact = _compact_text(first_paragraph)
        has_noise_term = any(_compact_text(term) in first_compact for term in noise_terms)
        if has_noise_term:
            paragraphs.pop(0)
            continue
        break

    return "\n\n".join(paragraphs).strip() or answer


def _normalize_generated_answer(answer: str, state: GraphState) -> str:
    normalized = _to_traditional_zh(answer)
    normalized = _remove_noise_opening(normalized, state["question"])
    return _normalize_partial_no_answer(normalized)


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
        answer_question = _build_answer_question(state)
        noise_terms = _extract_noise_terms(state["question"])
        noise_hint = ""
        if noise_terms:
            noise_hint = (
                "偵測到與知識庫無關的使用者自我介紹或助理身份猜測："
                f"{'、'.join(noise_terms)}。不要稱呼、模仿、承接或回答這些片段。\n"
            )

        rewrite_hint = ""
        if state.get("rewritten_question") and state["question"] != state["rewritten_question"]:
            rewrite_hint = f"檢索改寫參考：{state['rewritten_question']}\n"

        rag_prompt = (
            f"原始使用者輸入：{_to_traditional_zh(state['question'])}\n"
            f"主要回答問題：{answer_question}\n\n"
            f"{rewrite_hint}"
            f"{noise_hint}"
            "可用內部文件內容：\n"
            + "\n\n---\n\n".join(context_blocks)
            + "\n\n請用繁體中文回答，並遵守以下規則：\n"
            "1. 只根據可用內部文件內容回答，不要使用文件外的知識。\n"
            "2. 如果使用者一次問多個子問題，請逐一處理；文件能支持的子問題要正常回答。\n"
            "3. 不要開場寒暄，不要稱呼使用者，不要回應自我介紹、助理身份猜測或無意義片段。\n"
            "4. 對於文件沒有支持、與知識庫主題無關、或只是使用者自我介紹的片段，不要編造答案；"
            f"請簡短標示「{PARTIAL_NO_ANSWER_MESSAGE}」。\n"
            f"5. 只有在整個主要回答問題都找不到可用依據時，才只回覆「{NO_ANSWER_MESSAGE}」。\n"
            f"6. 如果已經回答了任何子問題，不要再輸出「{NO_ANSWER_MESSAGE}」。\n"
            "7. 回答最後列出引用來源。"
        )

    return {
        **state,
        "answer_question": _build_answer_question(state) if selected else "",
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
    gemini_resource_exhausted = False

    if state.get("use_gemini", True):
        gemini_model = state.get("gemini_model") or GEMINI_MODEL
        try:
            answer = generate_with_gemini(
                state.get("rag_prompt", ""),
                model=gemini_model,
                api_key=state.get("gemini_api_key") or None,
            )
            answer = _normalize_generated_answer(answer, state)
            return {
                **state,
                "answer": _append_sources(answer, sources),
                "answer_mode": "Gemini RAG",
                "gemini_status": f"Gemini API 已成功使用模型 {gemini_model} 生成回答。",
                "gemini_resource_exhausted": False,
                "steps": [*state.get("steps", []), step_name],
            }
        except GeminiApiError as exc:
            gemini_error = str(exc)
            if exc.resource_exhausted:
                gemini_resource_exhausted = True
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
            "gemini_resource_exhausted": gemini_resource_exhausted,
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
            "gemini_resource_exhausted": gemini_resource_exhausted,
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
        answer = _normalize_generated_answer(answer, state)
    except Exception as exc:
        return {
            **state,
            "answer": _build_extractive_answer(selected),
            "answer_mode": "Extractive RAG Fallback",
            "gemini_status": gemini_status,
            "gemini_error": gemini_error,
            "gemini_resource_exhausted": gemini_resource_exhausted,
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
        "gemini_resource_exhausted": gemini_resource_exhausted,
        "llm_status": _join_status(gemini_status, status.message),
        "steps": [*state.get("steps", []), step_name],
    }


def build_retrieval_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("rewrite_query_for_retrieval", rewrite_query_for_retrieval)
    workflow.add_node("retrieve_from_chroma", retrieve_from_chroma)
    workflow.add_node("rank_and_select_context", rank_and_select_context)
    workflow.add_node("build_rag_prompt", build_rag_prompt)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("rewrite_query_for_retrieval")
    workflow.add_edge("rewrite_query_for_retrieval", "retrieve_from_chroma")
    workflow.add_edge("retrieve_from_chroma", "rank_and_select_context")
    workflow.add_edge("rank_and_select_context", "build_rag_prompt")
    workflow.add_edge("build_rag_prompt", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_retrieval_graph():
    return build_retrieval_graph()


def build_query_state(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    use_gemini: bool = True,
    gemini_api_key: str = "",
    gemini_model: str = GEMINI_MODEL,
    use_llm: bool = True,
    llm_model: str = OLLAMA_MODEL,
    rewrite_query: bool = True,
) -> GraphState:
    return {
        "question": question,
        "top_k": top_k,
        "use_gemini": use_gemini,
        "gemini_api_key": gemini_api_key,
        "gemini_model": gemini_model,
        "use_llm": use_llm,
        "llm_model": llm_model,
        "rewrite_query": rewrite_query,
        "steps": [],
    }


def stream_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    use_gemini: bool = True,
    gemini_api_key: str = "",
    gemini_model: str = GEMINI_MODEL,
    use_llm: bool = True,
    llm_model: str = OLLAMA_MODEL,
    rewrite_query: bool = True,
):
    graph = get_retrieval_graph()
    initial_state = build_query_state(
        question=question,
        top_k=top_k,
        use_gemini=use_gemini,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        use_llm=use_llm,
        llm_model=llm_model,
        rewrite_query=rewrite_query,
    )

    for event in graph.stream(initial_state):
        for node_name, node_state in event.items():
            yield node_name, node_state


def run_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    use_gemini: bool = True,
    gemini_api_key: str = "",
    gemini_model: str = GEMINI_MODEL,
    use_llm: bool = True,
    llm_model: str = OLLAMA_MODEL,
    rewrite_query: bool = True,
) -> GraphState:
    graph = get_retrieval_graph()
    return graph.invoke(
        build_query_state(
            question=question,
            top_k=top_k,
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            use_llm=use_llm,
            llm_model=llm_model,
            rewrite_query=rewrite_query,
        )
    )
