# LLM Twin Project - Phases, Architecture & Roadmap

This document outlines the data requirements, data processing pipeline, fine-tuning strategy, advanced RAG engine, and step-by-step roadmap for building and deploying the **LLM Twin** system following the *LLM Engineer's Handbook* (Paul Iusztin & Maxime Labonne).

---

## 📁 1. Data Requirements

The LLM Twin models your personal voice, writing style, tone, and domain expertise.

### Sample Data Sources:
- **Personal Content:** LinkedIn posts, blog articles (Medium/Substack), tweets/threads, essays, technical docs, GitHub READMEs/notes, or personal PDFs.
- **Alternative Content (Synthetic/Curated):** If you don't have existing personal writing, curated articles or synthetic posts in your desired writing style (~20–50 articles or ~10,000–50,000 words total).

---

## ⚙️ 2. Data Processing Pipeline (ETL & Vector Indexing)

```
[Raw Content] ➔ [1. Text Cleaning] ➔ [2. Semantic Chunking] ➔ [3. Vector Embeddings] ➔ [4. Qdrant Cloud Storage]
```

1. **Text Cleaning (`cleaner.py`):** Removes HTML, normalizes spacing, extracts metadata (title, date, tags, URL).
2. **Semantic Chunking (`chunker.py`):** Splits text into ~300–500 word chunks with a 50-word overlap to preserve contextual continuity.
3. **Embeddings Generation (`vector_db.py`):** Converts chunks to 384-dimensional or 768-dimensional dense vectors (e.g. `all-MiniLM-L6-v2` or `bge-small-en-v1.5`).
4. **Vector Storage (Qdrant Cloud):** Stores vectors alongside metadata payloads for hybrid search and payload filtering.

---

## 🧠 3. Model Fine-Tuning (Instruction Tuning & QLoRA)

Fine-tuning teaches the base model (e.g., Llama 3 8B or Llama 3.2 3B) your sentence structure, vocabulary, and perspective.

1. **Instruction Dataset Creation (`dataset_generator.py`):** Converts raw text into JSON instruction-response pairs:
   ```json
   {
     "instruction": "Explain RAG pipelines in simple terms.",
     "response": "Here is how I think about RAG: Imagine you're taking an open-book exam..."
   }
   ```
2. **QLoRA Fine-Tuning (Google Colab Notebook):**
   - Executed on Google Colab using a free NVIDIA T4 GPU.
   - Utilizes `Unsloth` / `PEFT` for parameter-efficient QLoRA training in 15–20 minutes.
3. **Adapter Registry:**
   - Trained LoRA adapter weights (~50MB) are pushed directly to your **Hugging Face Model Hub** repository.

---

## 🔄 4. Advanced RAG Engine & Model Serving

1. **Query Expansion:** Expands user query into 2–3 alternative angles to maximize vector search recall.
2. **Filtered Search & Reranking:** Qdrant retrieves candidate chunks, and a lightweight reranker (`FlashRank`) orders the top 3 matches.
3. **Grounded Generation:** Sends user query + retrieved context + fine-tuned style prompt to **Groq Cloud API** / **Hugging Face Serverless Inference** for sub-second generation.

---

## 🗺️ 5. Implementation Roadmap

- [x] **Phase 1:** Project foundation, architecture design, and environment configuration.
- [ ] **Phase 2:** ETL Pipeline & Data Cleaning (`src/etl/`).
- [ ] **Phase 3:** Qdrant Vector DB Indexing & Embeddings (`src/database/`).
- [ ] **Phase 4:** Instruction Dataset & Colab Fine-Tuning Notebook (`notebooks/`).
- [ ] **Phase 5:** Advanced RAG Engine with Query Expansion & Reranking (`src/rag/`).
- [ ] **Phase 6:** Production Gate & Evaluation Suite (`src/evaluation/`).
- [ ] **Phase 7:** Gradio Web Interface & Automated Deployment to Hugging Face Spaces + GitHub (`app.py`, `.github/workflows/`).
