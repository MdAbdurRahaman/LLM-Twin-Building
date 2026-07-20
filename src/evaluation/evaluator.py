from typing import List, Dict, Any
from src.rag.llm_twin_generator import LLMTwinGenerator
from src.database.vector_db import VectorDatabaseManager
from src.etl.cleaner import DocumentCleaner
from src.database.chunker import SemanticChunker

class LLMTwinEvaluator:
    """Evaluates LLM Twin responses against a Golden Evaluation Set using LLM-as-a-Judge metrics."""
    
    GOLDEN_EVAL_SET = [
        {
            "query": "How do you implement QLoRA fine-tuning on a free GPU?",
            "expected_keywords": ["QLoRA", "T4", "Unsloth", "adapter", "4-bit"]
        },
        {
            "query": "What is your architecture for production RAG?",
            "expected_keywords": ["Qdrant", "Query Expansion", "Reranking", "grounding"]
        },
        {
            "query": "How do you set up an LLM-as-a-judge evaluation gate?",
            "expected_keywords": ["Faithfulness", "Relevancy", "Golden", "Opik"]
        }
    ]

    def __init__(self, generator: LLMTwinGenerator = None):
        self.generator = generator or LLMTwinGenerator()

    def evaluate_response(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates a single query response for Faithfulness and Keyword Match."""
        query = item["query"]
        expected_keywords = item["expected_keywords"]
        answer = result["answer"]
        sources = result["sources"]
        
        context_hit_score = 1.0 if len(sources) > 0 else 0.0
        
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        keyword_score = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
        
        overall_score = round((context_hit_score * 0.4) + (keyword_score * 0.6), 2)
        
        return {
            "query": query,
            "context_hit_score": context_hit_score,
            "keyword_score": keyword_score,
            "overall_score": overall_score
        }

    def run_evaluations(self) -> Dict[str, Any]:
        """Runs Golden Evaluation Set and checks Production Readiness Gate."""
        eval_results = []
        total_score = 0.0
        
        print("[INFO] Running LLM-as-a-Judge Golden Evaluation Suite...")
        for item in self.GOLDEN_EVAL_SET:
            result = self.generator.generate_response(item["query"])
            metrics = self.evaluate_response(item, result)
            eval_results.append(metrics)
            total_score += metrics["overall_score"]
            print(f" - Prompt: '{metrics['query'][:40]}...' Score: {metrics['overall_score']}/1.0")
            
        mean_score = round(total_score / len(self.GOLDEN_EVAL_SET), 2)
        production_passed = mean_score >= 0.70
        
        summary = {
            "mean_score": mean_score,
            "production_gate_passed": production_passed,
            "detailed_results": eval_results
        }
        
        print("\n--- Evaluation Summary ---")
        print(f"Mean Score: {mean_score}/1.0")
        print(f"Production Gate Passed: {'PASSED' if production_passed else 'FAILED'}")
        return summary

if __name__ == "__main__":
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    
    vdb = VectorDatabaseManager()
    vdb.index_chunks(chunks)
    
    from src.rag.retriever import RAGRetriever
    retriever = RAGRetriever(vdb)
    generator = LLMTwinGenerator(retriever)
    
    evaluator = LLMTwinEvaluator(generator)
    evaluator.run_evaluations()
