# LLM Twin - Personal AI Writing Character & Production RAG System

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://llm-twin-building-wnnn5ayppcpiler36kwtet.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?style=for-the-badge)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3-orange?style=for-the-badge)](https://groq.com/)

> An end-to-end production AI Character Twin built following the *LLM Engineer's Handbook* (Paul Iusztin & Maxime Labonne, Packt). The system learns to write like you by incorporating your personal writing style, voice, and domain expertise into an LLM grounded by vector RAG.

---

## 📌 Deliverables & Live URLs

- **GitHub Repository (Deliverable Submission):** [https://github.com/MdAbdurRahaman/LLM-Twin-Building](https://github.com/MdAbdurRahaman/LLM-Twin-Building)
- **Live Demo Web Application URL:** [https://llm-twin-building-wnnn5ayppcpiler36kwtet.streamlit.app](https://llm-twin-building-wnnn5ayppcpiler36kwtet.streamlit.app)
- **Model Architecture:** Groq Llama 3.3 70B + Qdrant Cloud Vector RAG + QLoRA Fine-Tuning Adapter

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 1. ETL PIPELINE                                   |
| Raw Writing (LinkedIn / Blog / Code / PDFs) -> Text Cleaner -> Semantic Chunker   |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+----------------------------------------+------------------------------------------+
|                              2. VECTOR INDEXING                                   |
| MiniLM Embeddings -> Qdrant Cloud Vector DB (Payload Filtered Vectors)            |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+----------------------------------------+------------------------------------------+
|                       3. QLORA FINE-TUNING (Colab)                                |
| Instruction Pairs -> Unsloth QLoRA -> Hugging Face Model Hub Adapter             |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+----------------------------------------+------------------------------------------+
|                        4. ADVANCED RAG & SERVING                                  |
| Query Expansion -> Qdrant Search -> Reranking -> Groq Llama-3 LLM Generation      |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+----------------------------------------+------------------------------------------+
|                     5. LIVE DEPLOYMENT (HF + GitHub)                              |
| Gradio Web UI on HF Spaces <--- Auto Sync via GitHub Actions <--- Git Push        |
+-----------------------------------------------------------------------------------+
```

---

## Production Pipeline Features

1. **ETL Data Pipeline (`src/etl/`):** Standardizes raw personal writing, strips noise, extracts tags/titles, and chunks documents semantically with sliding overlaps.
2. **Qdrant Vector Database (`src/database/`):** Computes dense vector embeddings using `all-MiniLM-L6-v2` and indexes payloads in Qdrant Cloud.
3. **QLoRA Fine-Tuning (`notebooks/`):** Ready-to-run Google Colab notebook fine-tuning Llama-3 on free T4 GPUs using Unsloth & PEFT.
4. **Advanced RAG Engine (`src/rag/`):** Features Multi-Query Expansion, score-based reranking, and grounded system prompts via Groq serverless inference.
5. **LLM-as-a-Judge Evaluation (`src/evaluation/`):** Automated test suite evaluating Faithfulness, Relevancy, and Style match with a production readiness threshold gate (0.70+).

---

## Quickstart Guide (Local Execution & Testing)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/llm-twin.git
cd llm-twin
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys (Optional for local testing—defaults to in-memory Qdrant and fallback mode):
```bash
cp .env.example .env
```

### 3. Run Ingestion & RAG Test
```bash
# Run LLM-as-a-Judge Evaluation Gate
python -m src.evaluation.evaluator
```

### 4. Launch Live Gradio Web UI
```bash
python app.py
```
Open your browser at `http://localhost:7860` to chat with your LLM Twin!

---

## License & Attribution
Reference implementation based on the book *LLM Engineer's Handbook* by Paul Iusztin & Maxime Labonne (Packt Publishing).
