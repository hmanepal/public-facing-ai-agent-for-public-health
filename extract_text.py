"""
Stage 1: Text Extraction
========================
Extracts text from:
  - Wikipedia articles (via the `wikipedia` library)
  - arXiv PDFs (via PyPDF2)

Saves raw text to .txt files in the `raw_text/` directory.

Install deps:
    pip install wikipedia-api PyPDF2
"""

import os
import re
import wikipediaapi
import PyPDF2

# ── Config ──────────────────────────────────────────────────────────────────

# Wikipedia articles to fetch  →  {output_filename: "Article Title"}
WIKIPEDIA_ARTICLES = {
    "wikipedia_covid19.txt": "COVID-19 pandemic",
    "wikipedia_who.txt": "World Health Organization",
}

# Local arXiv PDFs to extract  →  {output_filename: "path/to/file.pdf"}
ARXIV_PDFS = {
    "arxiv_paper1.txt": "data/arxiv_paper1.pdf",
}

OUTPUT_DIR = "raw_text"

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_wikipedia(title: str) -> str:
    """Fetch the full plain-text content of a Wikipedia article."""
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="PublicHealthRAG/1.0 (your@email.com)"  # required by Wikipedia API
    )
    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"Wikipedia page not found: '{title}'")
    return page.text


def extract_pdf_text(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyPDF2."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                print(f"  ⚠  Page {page_num + 1} returned no text (may be scanned/image-based)")

    return "\n".join(text_parts)


def save_text(text: str, filename: str) -> None:
    """Save extracted text to a file in OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  ✓  Saved → {filepath}  ({len(text):,} chars)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # --- Wikipedia ---
    print("\n📖  Fetching Wikipedia articles...")
    for filename, title in WIKIPEDIA_ARTICLES.items():
        print(f"  Fetching: '{title}'")
        try:
            text = fetch_wikipedia(title)
            save_text(text, filename)
        except Exception as e:
            print(f"  ✗  Failed ({title}): {e}")

    # --- arXiv PDFs ---
    print("\n📄  Extracting arXiv PDFs...")
    for filename, pdf_path in ARXIV_PDFS.items():
        print(f"  Extracting: '{pdf_path}'")
        try:
            text = extract_pdf_text(pdf_path)
            save_text(text, filename)
        except Exception as e:
            print(f"  ✗  Failed ({pdf_path}): {e}")

    print("\n✅  Extraction complete. Raw text files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()