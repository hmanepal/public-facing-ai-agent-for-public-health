"""
ingest.py — Incremental Ingest Pipeline (Stages 1–3)
=====================================================
Drop new PDFs into `data/` and run this script. It will:
  - Skip any source that has already been extracted + embedded
  - Only process new files
  - Merge new passages + embeddings into the existing DataFrame

Run:
    python ingest.py

Install deps:
    pip install wikipedia-api PyPDF2 pandas numpy sentence-transformers torch
"""

import os
import re
import json
import numpy as np
import pandas as pd
import wikipediaapi
import PyPDF2
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────

WIKIPEDIA_ARTICLES = {
    "wikipedia_covid19":         "COVID-19 pandemic",
    "wikipedia_who":             "World Health Organization",
    "wikipedia_cdc":             "Centers for Disease Control and Prevention",
    "wikipedia_vaccine_hesitancy": "Vaccine hesitancy",
    "wikipedia_herd_immunity":   "Herd immunity",
    "wikipedia_epidemiology":    "Epidemiology",
    "wikipedia_social_determinants": "Social determinants of health",
    "wikipedia_opioid_epidemic": "Opioid epidemic in the United States",
    "wikipedia_mental_health":   "Mental health",
    "wikipedia_antimicrobial_resistance": "Antimicrobial resistance",
    "wikipedia_global_health":   "Global health",
    "wikipedia_health_equity":   "Health equity",
}

PDF_DIR       = "data"
RAW_TEXT_DIR  = "raw_text"
PROCESSED_DIR = "processed"

PASSAGES_PKL  = os.path.join(PROCESSED_DIR, "passages.pkl")
PASSAGES_CSV  = os.path.join(PROCESSED_DIR, "passages.csv")
EMBEDDINGS_PKL = os.path.join(PROCESSED_DIR, "passages_with_embeddings.pkl")
EMBEDDINGS_NPY = os.path.join(PROCESSED_DIR, "embeddings_matrix.npy")
MANIFEST_PATH  = os.path.join(PROCESSED_DIR, "index_manifest.json")

TARGET_WORDS  = 150
MIN_WORDS     = 20
BATCH_SIZE    = 64
MODEL_NAME    = "all-MiniLM-L6-v2"

# ── Helpers: Stage 1 ─────────────────────────────────────────────────────────

def already_extracted(source_name: str) -> bool:
    """Return True if a .txt file already exists for this source."""
    return os.path.exists(os.path.join(RAW_TEXT_DIR, source_name + ".txt"))


def fetch_wikipedia(title: str) -> str:
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="PublicHealthRAG/1.0 (your@email.com)"
    )
    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"Wikipedia page not found: '{title}'")
    return page.text


def extract_pdf_text(pdf_path: str) -> str:
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                print(f"    ⚠  Page {page_num + 1} returned no text (scanned?)")
    return "\n".join(text_parts)


def save_raw_text(text: str, source_name: str) -> None:
    os.makedirs(RAW_TEXT_DIR, exist_ok=True)
    path = os.path.join(RAW_TEXT_DIR, source_name + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"    ✓  Saved raw text → {path}  ({len(text):,} chars)")


# ── Helpers: Stage 2 ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("\x0c", " ").replace("\xad", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" \n", "\n", text)
    return text.strip()


def split_into_chunks(text: str) -> list:
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= TARGET_WORDS:
            chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk, current_count = [], 0
            for sentence in sentences:
                sent_words = len(sentence.split())
                if current_count + sent_words > TARGET_WORDS and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_count = [sentence], sent_words
                else:
                    current_chunk.append(sentence)
                    current_count += sent_words
            if current_chunk:
                chunks.append(" ".join(current_chunk))
    return [c for c in chunks if len(c.split()) >= MIN_WORDS]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_TEXT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Load existing state ───────────────────────────────────────────────────

    if os.path.exists(EMBEDDINGS_PKL):
        existing_df = pd.read_pickle(EMBEDDINGS_PKL)
        existing_embeddings = np.load(EMBEDDINGS_NPY)
        already_processed = set(existing_df["source_doc"].unique())
        next_passage_id = existing_df["passage_id"].max() + 1
        print(f"\n📦  Loaded existing corpus: {len(existing_df):,} passages "
              f"from {len(already_processed)} source(s)")
    else:
        existing_df = pd.DataFrame(columns=["passage_id", "passage_text", "source_doc", "embedding"])
        existing_embeddings = None
        already_processed = set()
        next_passage_id = 0
        print("\n📦  No existing corpus found — building from scratch.")

    # ── Stage 1: Identify new sources ────────────────────────────────────────

    new_sources = []   # list of (source_name, raw_text)

    print("\n─── Stage 1: Extraction ────────────────────────────────────────────")

    # Wikipedia
    for source_name, title in WIKIPEDIA_ARTICLES.items():
        if source_name in already_processed:
            print(f"  ⏭  Skipping (already processed): {source_name}")
            continue
        if already_extracted(source_name):
            print(f"  📄  Loading cached raw text: {source_name}")
            with open(os.path.join(RAW_TEXT_DIR, source_name + ".txt"), "r", encoding="utf-8") as f:
                new_sources.append((source_name, f.read()))
        else:
            print(f"  🌐  Fetching Wikipedia: '{title}'")
            try:
                text = fetch_wikipedia(title)
                save_raw_text(text, source_name)
                new_sources.append((source_name, text))
            except Exception as e:
                print(f"  ✗  Failed: {e}")

    # PDFs
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) \
                if os.path.isdir(PDF_DIR) else []

    for pdf_filename in pdf_files:
        source_name = os.path.splitext(pdf_filename)[0]
        if source_name in already_processed:
            print(f"  ⏭  Skipping (already processed): {source_name}")
            continue
        if already_extracted(source_name):
            print(f"  📄  Loading cached raw text: {source_name}")
            with open(os.path.join(RAW_TEXT_DIR, source_name + ".txt"), "r", encoding="utf-8") as f:
                new_sources.append((source_name, f.read()))
        else:
            pdf_path = os.path.join(PDF_DIR, pdf_filename)
            print(f"  📄  Extracting PDF: '{pdf_filename}'")
            try:
                text = extract_pdf_text(pdf_path)
                save_raw_text(text, source_name)
                new_sources.append((source_name, text))
            except Exception as e:
                print(f"  ✗  Failed ({pdf_filename}): {e}")

    if not new_sources:
        print("\n✅  Nothing new to process. Corpus is up to date.")
        return

    print(f"\n  → {len(new_sources)} new source(s) to process: "
          f"{', '.join(s[0] for s in new_sources)}")

    # ── Stage 2: Preprocess new sources ──────────────────────────────────────

    print("\n─── Stage 2: Preprocessing ─────────────────────────────────────────")

    new_records = []
    for source_name, raw_text in new_sources:
        cleaned = clean_text(raw_text)
        chunks  = split_into_chunks(cleaned)
        print(f"  {source_name:<45} {len(chunks):>4} passages")
        for chunk in chunks:
            new_records.append({
                "passage_id":   next_passage_id,
                "passage_text": chunk,
                "source_doc":   source_name,
            })
            next_passage_id += 1

    new_df = pd.DataFrame(new_records, columns=["passage_id", "passage_text", "source_doc"])

    # ── Stage 3: Vectorize new passages ──────────────────────────────────────

    print(f"\n─── Stage 3: Vectorization ─────────────────────────────────────────")
    print(f"  🤖  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"  ⚙️   Encoding {len(new_df):,} new passages...")
    new_embeddings = model.encode(
        new_df["passage_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    new_df["embedding"] = list(new_embeddings)

    # ── Merge with existing corpus ────────────────────────────────────────────

    print("\n─── Merging ────────────────────────────────────────────────────────")

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    if existing_embeddings is not None:
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        combined_embeddings = new_embeddings

    # Save
    combined_df.to_pickle(EMBEDDINGS_PKL)
    combined_df[["passage_id", "passage_text", "source_doc"]].to_pickle(PASSAGES_PKL)
    combined_df[["passage_id", "passage_text", "source_doc"]].to_csv(PASSAGES_CSV, index=False)
    np.save(EMBEDDINGS_NPY, combined_embeddings)

    # Update manifest
    source_counts = combined_df.groupby("source_doc").size().to_dict()
    manifest = {
        "num_passages":    int(len(combined_df)),
        "embedding_dim":   int(combined_embeddings.shape[1]),
        "sources":         source_counts,
        "embeddings_path": EMBEDDINGS_NPY,
        "passages_path":   EMBEDDINGS_PKL,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Added  : {len(new_df):,} new passages")
    print(f"  Total  : {len(combined_df):,} passages across {len(source_counts)} source(s)")
    print(f"\n✅  Corpus updated. Ready to query with 05_rag_query.py")


if __name__ == "__main__":
    main()