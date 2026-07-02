import argparse

from app.config import DEFAULT_TOP_K, GEMINI_MODEL, OLLAMA_MODEL
from app.rag_graph import run_query


def search(
    query: str,
    k: int = DEFAULT_TOP_K,
    use_gemini: bool = True,
    gemini_model: str = GEMINI_MODEL,
    use_llm: bool = True,
    llm_model: str = OLLAMA_MODEL,
):
    response = run_query(
        query,
        top_k=k,
        use_gemini=use_gemini,
        gemini_model=gemini_model,
        use_llm=use_llm,
        llm_model=llm_model,
    )

    print("\nLangGraph steps")
    print(" -> ".join(response.get("steps", [])))
    print(f"\nAnswer mode: {response.get('answer_mode', 'N/A')}")
    if response.get("gemini_status"):
        print(f"Gemini status: {response['gemini_status']}")
    if response.get("llm_status"):
        print(f"LLM status: {response['llm_status']}")
    print("\nAnswer")
    print(response.get("answer", ""))
    if response.get("no_answer_reason"):
        print(f"\nNo answer reason: {response['no_answer_reason']}")

    selected = response.get("selected", [])
    if not selected:
        print("\n查無可用引用依據")
        return

    print("\nSelected documents")
    for item in selected:
        print("=" * 80)
        print(f"Rank: {item['rank']}")
        print(f"Distance: {item['distance']}")
        print(f"Match score: {item['confidence']}")
        print(f"Source: {item['source']}#chunk-{item['chunk_id']}")
        print(item["content"])


def parse_args():
    parser = argparse.ArgumentParser(description="Query the local LangGraph RAG flow.")
    parser.add_argument("question", nargs="?", help="Question to ask the knowledge base.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini API generation.")
    parser.add_argument("--gemini-model", default=GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument("--llm", action="store_true", help="Compatibility flag; local Ollama fallback is enabled by default.")
    parser.add_argument("--no-local-llm", action="store_true", help="Disable local Ollama fallback.")
    parser.add_argument("--llm-model", default=OLLAMA_MODEL, help="Ollama model name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    question = args.question or input("請輸入查詢問題：")
    search(
        question,
        k=args.top_k,
        use_gemini=not args.no_gemini,
        gemini_model=args.gemini_model,
        use_llm=not args.no_local_llm,
        llm_model=args.llm_model,
    )
