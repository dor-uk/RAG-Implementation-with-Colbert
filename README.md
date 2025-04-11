# RAG-Implementation-with-Colbert

A retrieval-augmented question-answering system built to provide accurate, context-aware responses from Turkish legal documents. The system processes legislation PDFs (e.g. 3308, 5580, 6721, 7528) using modern NLP techniques including ColBERT embeddings, Qdrant vector storage, and reranking with MiniLM.

---

Features
- Semantic Search over chunked legal texts.
- Reranking using Cross-Encoder (MiniLM) for better relevance.
- LLM-Powered Answering with GPT-like response generation.
- PDF Preprocessing with chunk merging, cleaning, and metadata enrichment.
- Fast Vector Search via in-memory Qdrant integration.

---
## What I used in the project

| Component        | Technology                                     |
|------------------|------------------------------------------------|
| Embedding Model  | `jinaai/jina-colbert-v2` via `fastembed`       |
| Vector DB        | `QdrantClient` (in-memory collection)          |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2`         |
| LLM              | `c4ai-command-r-plus-GPTQ` via proxy endpoint  |
| UI & Backend     | `Chainlit`, `Langchain`, `FastAPI`             |
| PDF Processing   | `PyPDF2`, `RecursiveCharacterTextSplitter`     |

---

## How to Run


1. **Parse legal documents**
   ```bash
   python parse_corpus.py
   ```

2. **Generate embeddings**
   ```bash
   python save_embeddings.py
   ```

3. **Run the Chainlit app**
   ```bash
   chainlit run app.py
   ```
