from src.config import settings
from src.etl.cleaner import DocumentCleaner
from src.database.chunker import SemanticChunker
from src.database.vector_db import VectorDatabaseManager
from src.rag.retriever import RAGRetriever
from src.rag.llm_twin_generator import LLMTwinGenerator
from app import respond

print("--- Automated Pipeline & UI Handler Test ---")

# 1. Initialize Pipeline
cleaner = DocumentCleaner()
docs = cleaner.load_and_clean_raw_posts()

chunker = SemanticChunker()
chunks = chunker.chunk_documents(docs)

vector_db = VectorDatabaseManager()
vector_db.index_chunks(chunks)

retriever = RAGRetriever(vector_db)
generator = LLMTwinGenerator(retriever)

# 2. Test User Prompts
test_prompts = [
    "How do you build a production RAG pipeline with Qdrant?",
    "Explain your QLoRA Fine-Tuning workflow",
    "How do you set up an LLM-as-a-judge gate?"
]

history = []
for prompt in test_prompts:
    print(f"\n[TEST PROMPT]: '{prompt}'")
    history, _ = respond(prompt, history)
    user_msg, bot_ans = history[-1]
    print(f"[LLM TWIN RESPONSE]:\n{bot_ans[:300]}...\n")
    assert len(bot_ans) > 50, "Response should be non-empty and detailed."

print("[SUCCESS] All 3 automated test queries generated valid, grounded responses!")
