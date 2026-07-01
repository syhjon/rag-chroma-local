import argparse
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
)
from app.embeddings import create_embedding_model


def load_documents(data_dir: Path = DATA_DIR):
    documents = []

    for file_path in sorted(data_dir.glob("*.txt")):
        documents.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={
                    "source": file_path.name,
                    "filename": file_path.name,
                    "path": str(file_path),
                },
            )
        )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    chunk_counts_by_source = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_counts_by_source[source] = chunk_counts_by_source.get(source, 0) + 1
        chunk.metadata["chunk_id"] = chunk_counts_by_source[source]
        chunk.metadata["chunk_chars"] = len(chunk.page_content)

    return chunks


def _reset_chroma_dir() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def build_vector_db(data_dir: Path = DATA_DIR, reset: bool = True):
    documents = load_documents(data_dir)

    if not documents:
        raise ValueError(f"{data_dir} 資料夾內沒有可匯入的 .txt 文件")

    chunks = split_documents(documents)

    embeddings = create_embedding_model()
    ids = [
        f"{chunk.metadata.get('source', 'unknown')}:{chunk.metadata.get('chunk_id', index)}"
        for index, chunk in enumerate(chunks, start=1)
    ]

    if reset:
        _reset_chroma_dir()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"讀取文件：{len(documents)} 份")
    print(f"匯入完成：{len(chunks)} 個 chunks")
    print(f"Chroma DB 位置：{CHROMA_DIR}")

    return vector_db


def parse_args():
    parser = argparse.ArgumentParser(description="Build the local Chroma vector DB.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append documents to the existing collection instead of rebuilding it.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing .txt files to ingest.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_vector_db(data_dir=args.data_dir, reset=not args.append)
