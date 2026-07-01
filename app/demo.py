from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
)
from app.ingest import build_vector_db
from app.local_llm import check_local_llm
from app.rag_graph import clear_runtime_caches, get_vector_db, run_query


st.set_page_config(
    page_title="Aegis Knowledge Core",
    page_icon=":material/hub:",
    layout="wide",
)


def get_collection_data():
    vector_db = get_vector_db()
    return vector_db._collection.get(include=["documents", "metadatas", "embeddings"])


def build_embedding_dataframe():
    data = get_collection_data()

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings = data.get("embeddings", [])

    if len(documents) == 0 or embeddings is None or len(embeddings) == 0:
        return pd.DataFrame()

    if len(embeddings) == 1:
        x_values = [0]
        y_values = [0]
    else:
        pca = PCA(n_components=2)
        points = pca.fit_transform(embeddings)
        x_values = points[:, 0]
        y_values = points[:, 1]

    rows = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        source = metadata.get("source", "unknown")
        chunk_id = metadata.get("chunk_id", index + 1)

        rows.append(
            {
                "index": index + 1,
                "x": x_values[index],
                "y": y_values[index],
                "source": source,
                "chunk_id": chunk_id,
                "preview": " ".join(document.split())[:160],
                "document": document,
            }
        )

    return pd.DataFrame(rows)


def list_source_files():
    return sorted(path.name for path in DATA_DIR.glob("*.txt"))


def display_answer_mode(mode: str) -> str:
    labels = {
        "Local LLM RAG": "本機 LLM RAG",
        "Extractive RAG": "來源式 RAG",
        "Extractive RAG Fallback": "來源式 RAG 備援",
        "No Context": "無可用內容",
    }
    return labels.get(mode, mode or "無資料")


def display_match_label(label: str) -> str:
    labels = {
        "High": "高",
        "Medium": "中",
        "Low": "低",
    }
    return labels.get(label, label or "無資料")


st.title("Aegis Knowledge Core")
st.caption("LangChain + LangGraph + Chroma 本機 RAG 知識庫展示")

with st.sidebar:
    st.header("知識庫設定")
    st.write(f"Collection：`{COLLECTION_NAME}`")
    st.write(f"向量資料庫：`{CHROMA_DIR.name}/`")
    st.write(f"Embedding 模型：`{EMBEDDING_MODEL.split('/')[-1]}`")

    st.divider()
    st.subheader("來源文件")
    files = list_source_files()
    if files:
        for filename in files:
            st.write(f"- `{filename}`")
    else:
        st.warning("data/ 目前沒有 .txt 文件")

    st.divider()
    st.subheader("本機 LLM")
    llm_model = st.text_input("Ollama 模型", value=OLLAMA_MODEL)

    llm_status = check_local_llm(model=llm_model)
    use_llm = st.toggle("使用 Ollama 生成回答", value=llm_status.available)
    if llm_status.available:
        st.success(llm_status.message)
    else:
        st.warning(llm_status.message)
        st.code(
            f"brew install ollama\nollama serve\nollama pull {llm_model}",
            language="bash",
        )

    st.divider()
    if st.button("重建 Chroma 索引", width="stretch"):
        with st.spinner("正在讀取文件、切分文件片段，並重建 Chroma..."):
            build_vector_db(reset=True)
            clear_runtime_caches()
        st.success("Chroma 索引已重建")
        st.rerun()


df = build_embedding_dataframe()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
with metric_1:
    st.metric("來源文件", len(list_source_files()))
with metric_2:
    st.metric("文件片段數量", len(df))
with metric_3:
    st.metric("流程節點", 4)
with metric_4:
    st.metric("執行模式", "Ollama" if use_llm else "本機檢索")

st.divider()

query_col, evidence_col = st.columns([1.05, 1.25], vertical_alignment="top")

with query_col:
    st.subheader("詢問知識庫")
    question = st.text_area(
        "問題",
        value="Aegis Knowledge Core 可以解決什麼問題？",
        height=92,
    )

    top_k = st.slider("取回文件數", min_value=1, max_value=6, value=4)
    run_button = st.button("執行 LangGraph 查詢", type="primary", width="stretch")

    st.markdown("#### LangGraph 查詢流程")
    st.code(
        "使用者問題 -> Chroma 檢索 -> 排序選取 -> 組合 RAG 提示詞 -> 本機 LLM / 備援回答",
        language="text",
    )

with evidence_col:
    st.subheader("RAG 回答")

    if run_button:
        with st.spinner("正在執行 LangGraph 檢索流程..."):
            response = run_query(
                question,
                top_k=top_k,
                use_llm=use_llm,
                llm_model=llm_model,
            )
    else:
        response = (
            run_query(
                question,
                top_k=top_k,
                use_llm=use_llm,
                llm_model=llm_model,
            )
            if not df.empty
            else {}
        )

    if not response:
        st.warning("Chroma 裡目前沒有資料。請先在側邊欄重建索引，或執行 python -m app.ingest。")
    else:
        st.markdown(response["answer"])

        result_metric_1, result_metric_2, result_metric_3, result_metric_4 = st.columns(4)
        with result_metric_1:
            st.metric("相符程度", display_match_label(response.get("confidence_label", "")))
        with result_metric_2:
            st.metric("取回數量", len(response.get("retrieved", [])))
        with result_metric_3:
            st.metric("選用數量", len(response.get("selected", [])))
        with result_metric_4:
            st.metric("回答模式", display_answer_mode(response.get("answer_mode", "")))

        if response.get("rag_prompt"):
            with st.expander("送給本機 LLM 的 RAG 提示詞", expanded=False):
                st.code(response["rag_prompt"], language="text")

        st.markdown("#### 引用依據")
        for item in response.get("retrieved", []):
            expanded = item["rank"] == 1
            with st.expander(
                f"排序 {item['rank']} | {item['source']}#chunk-{item['chunk_id']} | 距離 {item['distance']}",
                expanded=expanded,
            ):
                st.progress(min(max(item["confidence"], 0), 1), text=f"相符分數 {item['confidence']}")
                st.write(item["content"])

st.divider()

map_col, table_col = st.columns([1.25, 1], vertical_alignment="top")

with map_col:
    st.subheader("Chroma 向量地圖")

    if df.empty:
        st.warning("目前沒有 embeddings。請先重建 Chroma 索引。")
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="source",
            hover_data=["index", "source", "chunk_id", "preview"],
            text="index",
            title="Chroma Embeddings 的 2D PCA 投影",
            labels={
                "x": "PCA 維度 1",
                "y": "PCA 維度 2",
                "source": "來源文件",
                "index": "索引",
                "chunk_id": "片段編號",
                "preview": "內容預覽",
            },
        )
        fig.update_traces(textposition="top center", marker=dict(size=14, opacity=0.88))
        fig.update_layout(
            height=540,
            xaxis_title="PCA 維度 1",
            yaxis_title="PCA 維度 2",
            legend_title_text="來源文件",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, width="stretch")

with table_col:
    st.subheader("已索引文件片段")

    if df.empty:
        st.info("目前尚未建立索引文件片段。")
    else:
        display_df = df[["index", "source", "chunk_id", "preview"]].rename(
            columns={
                "index": "索引",
                "source": "來源文件",
                "chunk_id": "片段編號",
                "preview": "內容預覽",
            }
        )
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=540,
        )

st.divider()

st.markdown(
    """
#### 這個展示呈現的能力

- 使用 LangChain 讀取本機文件、切分文件片段、建立 embeddings，並串接 Chroma。
- 使用 Chroma 保存本機向量資料庫，不需要額外啟動外部資料庫服務。
- 使用 LangGraph 將檢索、排序、提示詞組合與回答生成拆成清楚節點。
- 使用 Streamlit 與 Plotly 同時展示 RAG 回答、引用依據與 embeddings 語意分布。
"""
)
