import json
from typing import List, Dict, Any
from pathlib import Path
from src.config import settings
from src.etl.cleaner import DocumentCleaner

class InstructionDatasetGenerator:
    """Generates instruction-response datasets for fine-tuning open-weight models (Llama 3/Qwen)."""
    
    def generate_instruction_pairs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts raw documents into instruction dataset items."""
        dataset = []
        
        for doc in docs:
            title = doc.get("title", "")
            content = doc.get("cleaned_content", "")
            
            item_1 = {
                "instruction": f"Can you explain your perspective on {title.lower()}?",
                "input": "",
                "output": content,
                "system": "You are an AI LLM Twin. You answer questions strictly reflecting the author's writing style, tone, and technical perspective."
            }
            dataset.append(item_1)
            
            item_2 = {
                "instruction": f"How do you approach {title} in production?",
                "input": "",
                "output": f"Here is my approach to {title}:\n\n{content}",
                "system": "You are an AI LLM Twin. You answer questions strictly reflecting the author's writing style, tone, and technical perspective."
            }
            dataset.append(item_2)
            
        return dataset

    def export_dataset(self, output_path: Path = None) -> Path:
        """Loads cleaned posts and exports instruction dataset JSON file."""
        if output_path is None:
            output_path = settings.PROCESSED_DATA_DIR / "instruction_dataset.json"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cleaner = DocumentCleaner()
        docs = cleaner.load_and_clean_raw_posts()
        dataset = self.generate_instruction_pairs(docs)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        print(f"[OK] Generated {len(dataset)} instruction pairs in '{output_path}'.")
        return output_path

if __name__ == "__main__":
    generator = InstructionDatasetGenerator()
    generator.export_dataset()
