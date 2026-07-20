from typing import List

class QueryExpander:
    """Generates expanded queries to improve vector database recall."""
    
    def expand_query(self, user_query: str) -> List[str]:
        """Generates multi-perspective query variations."""
        queries = [user_query]
        
        query_lower = user_query.lower()
        if "rag" in query_lower or "retrieval" in query_lower:
            queries.append(f"How to build production RAG vector search: {user_query}")
            queries.append(f"Vector database indexing chunking reranking {user_query}")
        elif "fine-tuning" in query_lower or "qlora" in query_lower or "training" in query_lower:
            queries.append(f"QLoRA fine tuning open weight model Unsloth: {user_query}")
            queries.append(f"Instruction dataset training LoRA adapter: {user_query}")
        elif "eval" in query_lower or "judge" in query_lower or "test" in query_lower:
            queries.append(f"LLM-as-a-judge evaluation frameworks Opik faithfulness: {user_query}")
        else:
            queries.append(f"Technical architecture and system design for {user_query}")
            
        return queries

if __name__ == "__main__":
    expander = QueryExpander()
    expanded = expander.expand_query("How do I evaluate my RAG pipeline?")
    print("Expanded Queries:")
    for q in expanded:
        print(f" - {q}")
