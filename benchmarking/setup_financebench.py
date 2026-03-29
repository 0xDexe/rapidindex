#!/usr/bin/env python3
# benchmarking/setup_financebench.py
"""
One-shot setup for the FinanceBench benchmark data.

Usage
-----
# Step 1 — download the JSONL question files only (fast, ~200 KB)
python -m benchmarking.setup_financebench

# Step 2 — also download all PDFs referenced by the questions (~2 GB, slow)
python -m benchmarking.setup_financebench --download-pdfs

# Step 3 — download only PDFs for a specific company (useful for quick smoke tests)
python -m benchmarking.setup_financebench --download-pdfs --company 3M

What gets created
-----------------
benchmarking/datasets/financebench/
  data/
    financebench_open_source.jsonl
    financebench_document_information.jsonl
  pdfs/
    3M_2018_10K.pdf
    ...

All data is sourced from the official public repository:
  https://github.com/patronus-ai/financebench
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Use urllib so there are zero extra dependencies at setup time.
# ---------------------------------------------------------------------------
import urllib.request
import urllib.error


BASE_GITHUB_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
BASE_GITHUB_PDF = "https://github.com/patronus-ai/financebench/raw/main/pdfs"

DATA_FILES = [
    "data/financebench_open_source.jsonl",
    "data/financebench_document_information.jsonl",
]

DEFAULT_DATA_DIR = Path("benchmarking/datasets/financebench")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Download url to dest with simple retry logic. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"  [{attempt}/{retries}] {url}")
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.HTTPError as exc:
            print(f"  HTTP {exc.code}: {exc.reason}")
            if exc.code == 404:
                return False          # no point retrying a 404
        except Exception as exc:
            print(f"  Error: {exc}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return False


def download_data_files(data_dir: Path) -> None:
    print("\n=== Downloading question/metadata JSONL files ===")
    for rel in DATA_FILES:
        dest = data_dir / rel
        if dest.exists():
            print(f"  Already present: {dest}")
            continue
        url = f"{BASE_GITHUB_RAW}/{rel}"
        ok = _download(url, dest)
        print("  OK" if ok else "  FAILED")


def load_doc_names(data_dir: Path, company_filter: Optional[str] = None) -> List[str]:
    """Return unique doc_names from the JSONL, optionally filtered by company."""
    questions_file = data_dir / "data" / "financebench_open_source.jsonl"
    if not questions_file.exists():
        print(f"Questions file not found: {questions_file}")
        sys.exit(1)

    doc_names = set()
    with open(questions_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if company_filter and row.get("company", "").upper() != company_filter.upper():
                continue
            if "doc_name" in row:
                doc_names.add(row["doc_name"])

    return sorted(doc_names)


def download_pdfs(data_dir: Path, company_filter: Optional[str] = None) -> None:
    doc_names = load_doc_names(data_dir, company_filter)

    label = f"company={company_filter}" if company_filter else "all companies"
    print(f"\n=== Downloading PDFs ({len(doc_names)} unique docs, {label}) ===")
    print("This may take a while — PDFs are 5–30 MB each.\n")

    pdfs_dir = data_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    success, failed = 0, []

    for doc_name in doc_names:
        dest = pdfs_dir / f"{doc_name}.pdf"
        if dest.exists():
            print(f"  Skip (exists): {doc_name}.pdf")
            success += 1
            continue

        url = f"{BASE_GITHUB_PDF}/{doc_name}.pdf"
        ok = _download(url, dest)
        if ok:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  OK ({size_mb:.1f} MB): {doc_name}.pdf")
            success += 1
        else:
            print(f"  FAILED: {doc_name}.pdf")
            failed.append(doc_name)
            # Remove partial file if it exists
            if dest.exists():
                dest.unlink()

    print(f"\nDone: {success} downloaded, {len(failed)} failed.")
    if failed:
        print("Failed docs:")
        for d in failed:
            print(f"  {d}")


def verify(data_dir: Path) -> None:
    """Print a quick status report."""
    print("\n=== Verification ===")

    for rel in DATA_FILES:
        p = data_dir / rel
        status = f"OK ({p.stat().st_size // 1024} KB)" if p.exists() else "MISSING"
        print(f"  {rel}: {status}")

    pdfs_dir = data_dir / "pdfs"
    pdf_count = len(list(pdfs_dir.glob("*.pdf"))) if pdfs_dir.exists() else 0
    print(f"  pdfs/: {pdf_count} files")

    questions_file = data_dir / "data" / "financebench_open_source.jsonl"
    if questions_file.exists():
        q_count = sum(1 for line in open(questions_file) if line.strip())
        print(f"  Questions: {q_count}")

        needed = set()
        with open(questions_file) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    needed.add(row.get("doc_name", ""))
        needed.discard("")

        present = {p.stem for p in pdfs_dir.glob("*.pdf")} if pdfs_dir.exists() else set()
        missing = needed - present
        print(f"  PDFs needed: {len(needed)}, present: {len(present)}, missing: {len(missing)}")
        if missing:
            print("  Missing PDFs (first 10):")
            for d in sorted(missing)[:10]:
                print(f"    {d}.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FinanceBench data for RapidIndex benchmarking"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Destination directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Also download PDF source documents (~2 GB total)",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Filter PDF downloads to a single company (e.g. 3M, AMAZON)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Just print a status report, do not download anything",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.verify:
        verify(data_dir)
        return

    download_data_files(data_dir)

    if args.download_pdfs:
        download_pdfs(data_dir, company_filter=args.company)

    verify(data_dir)
    print(
        "\nNext steps:\n"
        "  # Index documents and run benchmark\n"
        "  python -m benchmarking.run_benchmark --sample-size 20 --provider ollama\n"
        "  python -m benchmarking.run_benchmark --sample-size 150 --provider anthropic\n"
    )


if __name__ == "__main__":
    main()
