import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from src.config import settings

class VectorDatabaseManager:
    """Manages Qdrant Vector DB indexing, embedding computation, and similarity search."""
    
    def __init__(self, collection_name: str = settings.QDRANT_COLLECTION_NAME):
        self.collection_name = collection_name
        
        print(f"[INFO] Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
        if settings.QDRANT_URL and settings.QDRANT_API_KEY:
            print(f"[INFO] Connecting to Qdrant Cloud Cluster at {settings.QDRANT_URL}...")
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        else:
            print("[INFO] Qdrant Cloud credentials not provided. Using in-memory Qdrant client for testing.")
            self.client = QdrantClient(":memory:")
            
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates collection in Qdrant if it does not exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            print(f"[INFO] Creating Qdrant Collection '{self.collection_name}' (dim={settings.EMBEDDING_DIMENSION})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE
                )
            )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of strings."""
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexes a list of text chunks with metadata into Qdrant."""
        if not chunks:
            return 0
            
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=chunk["payload"]
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"[OK] Upserted {len(points)} vector chunks into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def search(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Searches vector DB for chunks most similar to the query."""
        query_vector = self.encoder.encode(query).tolist()
        
        try:
            # Modern qdrant-client API (v1.10+)
            query_res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            search_result = query_res.points
        except Exception:
            # Fallback for older API versions
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
        
        results = []
        for point in search_result:
            results.append({
                "score": round(point.score, 4),
                "title": point.payload.get("title", ""),
                "source_url": point.payload.get("source_url", ""),
                "tags": point.payload.get("tags", []),
                "text": point.payload.get("text", "")
            })
            
        return results

if __name__ == "__main__":
    from src.etl.cleaner import DocumentCleaner
    from src.database.chunker import SemanticChunker
    
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(docs)
    
    vdb = VectorDatabaseManager()
    vdb.index_chunks(chunks)
    
    test_query = "How do I build a production RAG pipeline?"
    search_res = vdb.search(test_query, top_k=2)
    print(f"\n[SEARCH] Query: '{test_query}'")
    for res in search_res:
        print(f" - [{res['score']}] {res['title']}: {res['text'][:100]}...")
