# Aegis Knowledge Core

這是一個可以在 MacBook 本機執行的 RAG 知識庫展示專案。專案整合 LangChain、LangGraph、Chroma、HuggingFace Embeddings、Gemini API、Ollama 本機 LLM 與 Streamlit，用來展示文件向量化、語意檢索、RAG 提示詞、雲端與本機 LLM 備援、來源引用與向量視覺化。

## 快速啟動展示

建議使用 Python 3.12。若 Mac 尚未安裝 Python，可先用 Homebrew 安裝：

```bash
brew install python@3.12
```

進入專案資料夾：

```bash
cd rag-chroma-local
```

建立並啟用虛擬環境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

安裝 Python 套件：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

建立或重建 Chroma 向量資料庫：

```bash
python -m app.ingest
```

啟動網頁展示：

```bash
streamlit run app/demo.py
```

瀏覽器打開 Streamlit 顯示的網址，通常是：

```text
http://localhost:8501
```

## 啟用 Gemini API

系統會優先使用 Gemini API 產生 RAG 回答。請先到 Google AI Studio 建立 API key，然後在終端機設定環境變數：

```bash
export GEMINI_API_KEY="你的 Gemini API Key"
```

也可以在專案根目錄建立 `.env`，內容如下：

```bash
GEMINI_API_KEY="你的 Gemini API Key"
```

`.env` 已被 `.gitignore` 排除，不會被送進 Git。預設 Gemini 模型設定在 [app/config.py](/Users/charlie_hsu/Documents/rag-chroma-local/app/config.py)：

```python
GEMINI_MODEL = "gemini-2.0-flash"
```

請不要把 API key 寫進 `app/config.py`。`GEMINI_API_KEY_ENV` 代表環境變數名稱，應維持為 `"GEMINI_API_KEY"`。

若 Gemini API 回傳 `429 RESOURCE_EXHAUSTED`，系統會自動切換到本機 Ollama LLM。

## 啟用本機 LLM 備援

如果只要展示 Chroma 檢索與 LangGraph 流程，可以跳過本段。若要在 Gemini 額度用完或不可用時使用本機 LLM 備援，請安裝 Ollama 並下載模型：

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

確認模型已安裝：

```bash
ollama list
```

預設模型設定在 [app/config.py](/Users/charlie_hsu/Documents/rag-chroma-local/app/config.py)：

```python
OLLAMA_MODEL = "llama3.2:3b"
```

啟動 Streamlit 後，側邊欄會顯示 Gemini API 與本機 LLM 狀態。若 Ollama 模型已就緒，可以開啟「使用 Ollama 作為備援」，當 Gemini 額度用完或不可用時，回答模式會切換為「本機 LLM RAG」。

## CLI 快速測試

不用開網頁，也可以直接在終端機查詢。預設會先嘗試 Gemini API，再嘗試本機 Ollama 備援，最後才使用來源式 RAG：

```bash
python -m app.query "Aegis Knowledge Core 可以解決什麼問題？"
```

停用 Gemini API：

```bash
python -m app.query "Aegis Knowledge Core 可以解決什麼問題？" --no-gemini
```

停用本機 LLM 備援：

```bash
python -m app.query "Aegis Knowledge Core 可以解決什麼問題？" --no-local-llm
```

指定取回文件數：

```bash
python -m app.query "這個系統用了哪些技術？" --top-k 5
```

## 專案結構

```text
rag-chroma-local/
├── app/
│   ├── config.py       # 路徑、collection、embedding model 與文件片段設定
│   ├── embeddings.py   # HuggingFace embedding factory 與本機 cache 設定
│   ├── gemini_llm.py   # Gemini API REST 呼叫與 429 RESOURCE_EXHAUSTED 判斷
│   ├── ingest.py       # 建立或重建 Chroma 向量資料庫
│   ├── local_llm.py    # Ollama 狀態檢查與 ChatOllama 呼叫
│   ├── rag_graph.py    # LangGraph RAG 查詢流程
│   ├── query.py        # CLI 查詢入口
│   └── demo.py         # Streamlit 視覺化展示
├── data/               # RAG 友善結構化純文字文件
├── chroma_db/          # 本機 Chroma persistent DB，執行 ingest 後產生，不納入 Git
├── .hf-cache/          # HuggingFace model cache，第一次執行後產生，不納入 Git
├── requirements.txt
└── README.md
```

## 展示重點

- LangChain：讀取 `data/` 文件、切分文件片段、建立 embeddings，並串接 Chroma vector store。
- Chroma：保存文件內容、metadata 與 embedding 向量，提供語意相似度搜尋。
- LangGraph：把查詢流程拆成檢索、排序、提示詞組合與回答生成節點。
- Gemini API：優先生成 RAG 回答，適合展示雲端 LLM 串接能力。
- Ollama：在本機執行 LLM，作為 Gemini 額度用完或不可用時的備援。
- Streamlit：展示查詢、回答模式、引用依據、RAG 提示詞與 Chroma 向量地圖。
- Plotly + PCA：將高維 embeddings 壓縮成 2D 圖，方便使用者理解語意分布。

## RAG 文件格式

`data/` 內的文件已整理成 RAG 友善格式。每份文件都包含：

- `metadata`
- `summary`
- `key_points`
- `content`
- `qa_examples`
- `retrieval_notes`

目前包含：

- `product_overview.txt`：產品定位、功能總覽、解決問題
- `business_value.txt`：商業價值、企業導入價值、專案展示亮點
- `system_architecture.txt`：系統流程、RAG 架構、資料流
- `technical_stack.txt`：LangChain、LangGraph、Chroma、LLM、Streamlit
- `use_cases.txt`：使用情境、查詢範例、展示腳本
- `enterprise_management.txt`：企業管理、導入情境與治理觀點
- `faq.txt`：常見問答
- `project_highlights.txt`：專案亮點與技術展示說法
- `troubleshooting.txt`：安裝、模型與錯誤排除

新增或修改文件後，請重新建立 Chroma index：

```bash
python -m app.ingest
```

## LangGraph 流程

目前 graph 定義在 [app/rag_graph.py](/Users/charlie_hsu/Documents/rag-chroma-local/app/rag_graph.py)，流程如下：

```text
使用者問題
  -> retrieve_from_chroma
  -> rank_and_select_context
  -> build_rag_prompt
  -> generate_answer
```

各節點職責：

- `retrieve_from_chroma`：使用 Chroma similarity search 取回相關文件片段。
- `rank_and_select_context`：依距離與排序選出要進入回答的 context。
- `build_rag_prompt`：將使用者問題與取回內容組成 RAG 提示詞。
- `generate_answer`：若有相近資料，先呼叫 Gemini API；若 Gemini 回傳 `429 RESOURCE_EXHAUSTED` 或不可用，切換到 Ollama 本機 LLM；若本機 LLM 不可用，改用來源式 RAG 備援回答；若沒有相近資料，回答「目前資料庫無此答案，請問其他問題」。

## 分享或交付前

若使用 Git 交付，建議版本控制下列內容：

- `app/`
- `data/`
- `requirements.txt`
- `README.md`
- `.gitignore`

不建議交付 `.venv/`，因為虛擬環境通常不適合跨電腦搬移。`chroma_db/` 可由 `python -m app.ingest` 重建，因此不納入 Git。`.hf-cache/` 可以不包含，收件人第一次執行時會自動下載；如果現場網路不穩，也可以另外附上 `.hf-cache/`，但檔案會比較大。

Ollama 模型預設存在使用者家目錄的 `~/.ollama/`，不在專案資料夾內。若收件人的 MacBook 尚未安裝模型，請依照「啟用本機 LLM 備援」步驟下載。

## 常見問題

### 第一次執行很慢

第一次安裝與匯入時，HuggingFace embedding model 需要下載到 `.hf-cache/`。若啟用本機 LLM，Ollama 也需要下載模型。下載完成後，後續啟動會快很多。

### Chroma 查不到資料

請確認 `data/` 內有 `.txt` 文件，然後重新執行：

```bash
python -m app.ingest
```

### Streamlit 找不到套件

請確認已啟用虛擬環境：

```bash
source .venv/bin/activate
```

再確認 Streamlit 路徑：

```bash
which streamlit
```

路徑應該要指向專案底下的 `.venv/bin/streamlit`。

### Gemini API 顯示未設定

請確認已設定 `GEMINI_API_KEY`。可以在終端機使用 `export GEMINI_API_KEY="..."`，或在專案根目錄建立 `.env`。

### Gemini API 額度用完

若 Gemini API 回傳 `429 RESOURCE_EXHAUSTED`，系統會自動切換到本機 Ollama LLM。若本機 LLM 也不可用，系統會改用來源式 RAG 備援回答。

### 本機 LLM 顯示未就緒

請確認 Ollama service 已啟動：

```bash
brew services start ollama
```

再確認模型已下載：

```bash
ollama list
```

若清單內沒有 `llama3.2:3b`，請執行：

```bash
ollama pull llama3.2:3b
```

## 展示建議

1. 打開 `data/`，說明文件已整理成 RAG 友善格式。
2. 執行 `python -m app.ingest`，展示文件如何被切成片段、embedding、寫入 Chroma。
3. 執行 `streamlit run app/demo.py`，展示自然語言查詢。
4. 在側邊欄確認 Gemini API 狀態，展示回答模式優先使用「Gemini RAG」。
5. 若要展示備援流程，可以停用 Gemini 或用完額度後觀察系統切換到「本機 LLM RAG」或「來源式 RAG 備援」。
6. 打開「送給 Gemini / 本機 LLM 的 RAG 提示詞」，說明 LLM 是根據 Chroma 取回內容回答。
7. 點開「引用依據」，說明回答可以追蹤到來源文件與文件片段。
8. 指向 Chroma 向量地圖，說明 embeddings 如何形成語意分布。
9. 打開 [app/rag_graph.py](/Users/charlie_hsu/Documents/rag-chroma-local/app/rag_graph.py)，說明未來可替換 LLM、加入 reranker、權限控管或 human-in-the-loop。
