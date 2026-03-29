# benchmarking/datasets/financebench_loader.py
"""
FinanceBench dataset loader.

The open-source release lives at:
  https://github.com/patronus-ai/financebench

Expected directory layout (created by setup_financebench.py):

  benchmarking/datasets/financebench/
    data/
      financebench_open_source.jsonl        # 150 Q&A rows
      financebench_document_information.jsonl   # PDF metadata / download URLs
    pdfs/
      3M_2018_10K.pdf
      AMAZON_2022_10K.pdf
      ...

Actual JSONL row shape
----------------------
{
  "financebench_id":    "financebench_id_03029",
  "company":            "3M",
  "doc_name":           "3M_2018_10K",          # matches PDF filename (no .pdf)
  "question_type":      "metrics-generated",
  "question_reasoning": "Information extraction",
  "question":           "What is the FY2018 capital expenditure ...",
  "answer":             "$1577.00",
  "justification":      "...",
  "dataset_subset_label": "OPEN_SOURCE",
  "evidence": [
    {
      "evidence_text":          "...",
      "doc_name":               "3M_2018_10K",
      "evidence_page_num":      47,
      "evidence_text_full_page": "..."
    }
  ]
}
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger


class FinanceBenchDataset:
    """FinanceBench dataset loader that matches the actual JSONL schema."""

    # Question categories — useful for slicing results by difficulty
    QUESTION_TYPES = {
        "metrics-generated":  "Fact extraction (direct number lookup)",
        "domain-relevant":    "Domain reasoning (ratios, interpretation)",
        "novel-generated":    "Open-ended reasoning",
    }

    def __init__(
        self,
        data_dir: str = "./benchmarking/datasets/financebench",
        subset: str = "OPEN_SOURCE",   # or "ALL"
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.questions: List[Dict] = []
        self.documents: List[Path] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[List[Dict], List[Path]]:
        """Load questions and locate PDF paths.

        Returns
        -------
        questions : list of dicts, each with keys:
            question, answer, doc_name, company, question_type,
            question_reasoning, evidence, financebench_id
        documents : list of Path objects pointing to PDF files
        """
        self.questions = self._load_questions()
        self.documents = self._locate_pdfs()

        logger.info(
            f"FinanceBench loaded: {len(self.questions)} questions, "
            f"{len(self.documents)} PDFs"
        )
        return self.questions, self.documents

    def get_sample(self, n: int = 10, question_type: str = None) -> List[Dict]:
        """Return up to n questions, optionally filtered by type."""
        qs = self.questions
        if question_type:
            qs = [q for q in qs if q.get("question_type") == question_type]
        return qs[:n]

    def get_pdf_for_question(self, question: Dict) -> Path | None:
        """Return the PDF path for a question, or None if not downloaded."""
        doc_name = question.get("doc_name", "")
        pdf_path = self.data_dir / "pdfs" / f"{doc_name}.pdf"
        return pdf_path if pdf_path.exists() else None

    def stats(self) -> Dict:
        """Quick summary of the loaded dataset."""
        from collections import Counter
        type_counts = Counter(q.get("question_type") for q in self.questions)
        company_counts = Counter(q.get("company") for q in self.questions)
        return {
            "total_questions": len(self.questions),
            "total_pdfs": len(self.documents),
            "by_type": dict(type_counts),
            "companies": len(company_counts),
            "top_companies": company_counts.most_common(5),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_questions(self) -> List[Dict]:
        questions_file = self.data_dir / "data" / "financebench_open_source.jsonl"

        if not questions_file.exists():
            raise FileNotFoundError(
                f"Questions file not found: {questions_file}\n"
                f"Run:  python -m benchmarking.setup_financebench"
            )

        rows = []
        with open(questions_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Filter to requested subset
                if self.subset != "ALL" and row.get("dataset_subset_label") != self.subset:
                    continue
                rows.append(row)

        if not rows:
            raise ValueError(
                f"No questions found for subset='{self.subset}' in {questions_file}"
            )

        logger.info(f"Loaded {len(rows)} questions (subset={self.subset})")
        return rows

    def _locate_pdfs(self) -> List[Path]:
        """Return paths for every PDF that actually exists on disk."""
        pdfs_dir = self.data_dir / "pdfs"
        if not pdfs_dir.exists():
            logger.warning(
                f"PDFs directory missing: {pdfs_dir}\n"
                f"Run:  python -m benchmarking.setup_financebench"
            )
            return []

        # Deduplicate: multiple questions can reference the same PDF
        needed_docs = {q["doc_name"] for q in self.questions if "doc_name" in q}
        found = []
        missing = []

        for doc_name in sorted(needed_docs):
            path = pdfs_dir / f"{doc_name}.pdf"
            if path.exists():
                found.append(path)
            else:
                missing.append(doc_name)

        if missing:
            logger.warning(
                f"{len(missing)} PDFs referenced in questions but not on disk:\n"
                + "\n".join(f"  {d}" for d in missing[:10])
                + ("\n  ..." if len(missing) > 10 else "")
                + f"\nRun:  python -m benchmarking.setup_financebench --download-pdfs"
            )

        logger.info(f"Located {len(found)} PDFs on disk ({len(missing)} missing)")
        return found


__all__ = ["FinanceBenchDataset"]
