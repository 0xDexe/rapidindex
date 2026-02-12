# benchmarks/datasets/financebench_loader.py
"""FinanceBench dataset loader."""

import json
from pathlib import Path
from typing import List, Dict
from loguru import logger


class FinanceBenchDataset:
    """FinanceBench dataset for benchmarking."""
    
    def __init__(self, data_dir: str = "./benchmarks/datasets/financebench"):
        """Initialize dataset."""
        self.data_dir = Path(data_dir)
        self.questions = []
        self.documents = []
        
    def load(self):
        """Load FinanceBench data."""
        # Load questions
        questions_file = self.data_dir / "questions.json"
        if questions_file.exists():
            with open(questions_file) as f:
                self.questions = json.load(f)
        
        # Load documents
        docs_dir = self.data_dir / "documents"
        if docs_dir.exists():
            self.documents = list(docs_dir.glob("*.pdf"))
        
        logger.info(f"Loaded {len(self.questions)} questions, {len(self.documents)} documents")
        
        return self.questions, self.documents
    
    def get_sample(self, n: int = 10):
        """Get sample questions."""
        return self.questions[:n]


__all__ = ['FinanceBenchDataset']