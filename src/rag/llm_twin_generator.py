from typing import List, Dict, Any
import os
from src.config import settings
from src.rag.retriever import RAGRetriever

class LLMTwinGenerator:
    """Generates persona-grounded responses using RAG context and LLM inference."""
    
    def __init__(self, retriever: RAGRetriever = None):
        self.retriever = retriever or RAGRetriever()
        self.groq_api_key = settings.GROQ_API_KEY
        
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("[INFO] Initialized Groq API client for LLM Twin generation.")
            except Exception as e:
                print(f"[WARN] Groq client initialization warning: {e}")
                self.groq_client = None
        else:
            self.groq_client = None
            print("[INFO] Groq API key not provided. Generator will operate in local fallback mode.")

    def format_system_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Constructs persona system prompt grounded in retrieved context."""
        context_str = "\n\n".join([
            f"--- Article Snippet [{i+1}] ({c['title']}) ---\n{c['text']}"
            for i, c in enumerate(context_chunks)
        ])
        
        system_prompt = (
            "You are an AI Character LLM Twin. You speak, write, and answer technical questions strictly "
            "incorporating the author's writing style, technical voice, and personal experience.\n\n"
            "GUIDELINES:\n"
            "1. Ground your answer in the retrieved context snippets provided below.\n"
            "2. Adopt a practical, concise, and technical tone.\n"
            "3. Use structured markdown formatting with bullet points where appropriate.\n"
            "4. If the retrieved context doesn't fully answer the query, draw upon sound AI engineering principles while preserving the author's voice.\n\n"
            f"RETRIEVED AUTHOR CONTEXT:\n{context_str if context_str else 'No specific articles retrieved.'}"
        )
        return system_prompt

    def generate_response(self, user_query: str) -> Dict[str, Any]:
        """Processes user query through RAG pipeline and generates LLM response."""
        context_chunks = self.retriever.retrieve_context(user_query)
        system_prompt = self.format_system_prompt(context_chunks)
        
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )
                answer_text = response.choices[0].message.content
            except Exception as e:
                print(f"[WARN] Groq API call failed: {e}. Falling back to local generation.")
                answer_text = self._fallback_response(user_query, context_chunks)
        else:
            answer_text = self._fallback_response(user_query, context_chunks)
            
        return {
            "query": user_query,
            "answer": answer_text,
            "sources": [
                {"title": c["title"], "score": c["score"], "source_url": c.get("source_url", "")}
                for c in context_chunks
            ]
        }

    def _fallback_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generates a structured fallback response when Groq API key is not active."""
        if not context_chunks:
            return (
                "**LLM Twin Response:**\n\n"
                "I couldn't find specific articles from my past writing matching your question. "
                "However, as an AI Engineer twin, I recommend starting with semantic chunking, Qdrant vector indexing, and instruction tuning!"
            )
            
        top_chunk = context_chunks[0]
        sources_list = "\n".join([f"- **{c['title']}** (Relevance Score: {c['score']})" for c in context_chunks])
        
        return (
            f"**LLM Twin Response (Grounded in Past Writing):**\n\n"
            f"Based on my article *\"{top_chunk['title']}\"*, here is how I approach this:\n\n"
            f"> \"{top_chunk['text'][:350]}...\"\n\n"
            f"**Key Takeaways:**\n"
            f"1. Always ground your system prompts in vector context to prevent hallucinations.\n"
            f"2. Use hybrid search and reranking for maximum precision.\n"
            f"3. Evaluate faithfulness using an LLM-as-a-judge production gate.\n\n"
            f"**Referenced Articles:**\n{sources_list}"
        )

if __name__ == "__main__":
    from src.etl.cleaner import DocumentCleaner
    from src.database.chunker import SemanticChunker
    from src.database.vector_db import VectorDatabaseManager
    
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    
    vdb = VectorDatabaseManager()
    vdb.index_chunks(chunks)
    
    retriever = RAGRetriever(vdb)
    generator = LLMTwinGenerator(retriever)
    
    result = generator.generate_response("How do I build production RAG?")
    print("\n--- LLM Twin Answer ---")
    print(result["answer"])
