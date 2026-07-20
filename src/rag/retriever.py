from typing import List, Dict, Any
from src.database.vector_db import VectorDatabaseManager
from src.rag.query_expansion import QueryExpander
from src.config import settings

class RAGRetriever:
    """Handles query expansion, multi-query vector search, and reranking."""
    
    def __init__(self, vector_db: VectorDatabaseManager = None):
        self.vector_db = vector_db or VectorDatabaseManager()
        self.expander = QueryExpander()

    def retrieve_context(self, user_query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Retrieves and reranks relevant context chunks for a user query."""
        expanded_queries = self.expander.expand_query(user_query)
        
        seen_texts = set()
        retrieved_chunks = []
        
        for q in expanded_queries:
            results = self.vector_db.search(q, top_k=top_k)
            for res in results:
                if res["text"] not in seen_texts:
                    seen_texts.add(res["text"])
                    retrieved_chunks.append(res)
                    
        retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = retrieved_chunks[:top_k]
        return top_chunks

if __name__ == "__main__":
    from src.etl.cleaner import DocumentCleaner
    from src.database.chunker import SemanticChunker
    
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    
    vdb = VectorDatabaseManager()
    vdb.index_chunks(chunks)
    
    retriever = RAGRetriever(vdb)
    context = retriever.retrieve_context("Tell me about QLoRA fine-tuning")
    print(f"Retrieved {len(context)} chunks:")
    for c in context:
        print(f" - [{c['score']}] {c['title']}")
