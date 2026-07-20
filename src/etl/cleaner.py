import json
import re
from typing import List, Dict, Any
from pathlib import Path
from src.config import settings

class DocumentCleaner:
    """Cleans and standardizes raw text documents for ETL pipeline."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Removes extra whitespace, HTML tags, and normalizes text."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('\xa0', ' ').replace('\r\n', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def load_and_clean_raw_posts(self, file_path: Path = None) -> List[Dict[str, Any]]:
        """Loads raw JSON posts and returns cleaned document list."""
        if file_path is None:
            file_path = settings.RAW_DATA_DIR / "sample_posts.json"
            
        if not file_path.exists():
            raise FileNotFoundError(f"Raw post dataset not found at {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw_posts = json.load(f)
            
        cleaned_docs = []
        for post in raw_posts:
            cleaned_content = self.clean_text(post.get("content", ""))
            cleaned_title = self.clean_text(post.get("title", ""))
            
            cleaned_doc = {
                "doc_id": post.get("id", ""),
                "title": cleaned_title,
                "author": post.get("author", "Author"),
                "date": post.get("date", ""),
                "tags": post.get("tags", []),
                "source_url": post.get("source_url", ""),
                "cleaned_content": cleaned_content,
                "full_text": f"Title: {cleaned_title}\nTags: {', '.join(post.get('tags', []))}\n\n{cleaned_content}"
            }
            cleaned_docs.append(cleaned_doc)
            
        return cleaned_docs

if __name__ == "__main__":
    cleaner = DocumentCleaner()
    docs = cleaner.load_and_clean_raw_posts()
    print(f"[OK] Successfully cleaned {len(docs)} sample posts.")
