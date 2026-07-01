import argparse

from app.config import DEFAULT_TOP_K, OLLAMA_MODEL
from app.rag_graph import run_query


def search(
    query: str,
    k: int = DEFAULT_TOP_K,
    use_llm: bool = False,
    llm_model: str = OLLAMA_MODEL,
):
    response = run_query(query, top_k=k, use_llm=use_llm, llm_model=llm_model)

    print("\nLangGraph steps")
    print(" -> ".join(response.get("steps", [])))
    print(f"\nAnswer mode: {response.get('answer_mode', 'N/A')}")
    if response.get("llm_status"):
        print(f"LLM status: {response['llm_status']}")
    print("\nAnswer")
    print(response.get("answer", ""))

    retrieved = response.get("retrieved", [])
    if not retrieved:
        print("\n查無相關結果")
        return

    print("\nRetrieved documents")
    for item in retrieved:
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
    parser.add_argument("--llm", action="store_true", help="Use local Ollama LLM generation.")
    parser.add_argument("--llm-model", default=OLLAMA_MODEL, help="Ollama model name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    question = args.question or input("請輸入查詢問題：")
    search(question, k=args.top_k, use_llm=args.llm, llm_model=args.llm_model)
