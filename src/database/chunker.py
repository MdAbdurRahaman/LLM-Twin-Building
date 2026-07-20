from typing import List, Dict, Any
from src.config import settings

class SemanticChunker:
    """Splits cleaned documents into overlapping semantic chunks for vector indexing."""
    
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits a single document into chunk dictionaries with metadata."""
        content = doc.get("cleaned_content", "")
        words = content.split()
        
        if not words:
            return []
            
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = f"{doc['doc_id']}_chunk_{chunk_num}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "author": doc["author"],
                "tags": doc["tags"],
                "source_url": doc["source_url"],
                "text": chunk_text,
                "chunk_index": chunk_num,
                "payload": {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "author": doc["author"],
                    "source_url": doc["source_url"],
                    "tags": doc["tags"],
                    "text": chunk_text
                }
            })
            
            chunk_num += 1
            start_idx += (self.chunk_size - self.overlap)
            if start_idx >= len(words):
                break
                
        return chunks

    def chunk_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunks a list of documents."""
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

if __name__ == "__main__":
    from src.etl.cleaner import DocumentCleaner
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks from {len(docs)} documents.")
