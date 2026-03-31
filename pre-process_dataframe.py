"""
Stage 2: Preprocessing & Structuring
=====================================
Reads .txt files from `raw_text/`, cleans and chunks the text into passages,
then stores them in a pandas DataFrame saved as `passages.csv` and `passages.pkl`.

DataFrame columns:
    passage_id    – unique integer ID
    passage_text  – cleaned text chunk
    source_doc    – source filename (without extension)

Install deps:
    pip install pandas
"""

import os
import re
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────

INPUT_DIR  = "raw_text"
OUTPUT_DIR = "processed"

# Target chunk size in words. Passages will split on paragraph breaks
# and then further subdivide if a paragraph exceeds this.
TARGET_WORDS = 150
MIN_WORDS    = 20      # discard passages shorter than this

# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize raw extracted text:
      - Collapse runs of whitespace/newlines
      - Remove common PDF artefacts (ligatures, soft-hyphens, form feeds)
      - Strip citation brackets like [1], [23]
      - Normalize unicode punctuation to ASCII equivalents
    """
    # PDF artefacts
    text = text.replace("\x0c", " ")          # form feed
    text = text.replace("\xad", "")           # soft hyphen
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures

    # Wikipedia / arXiv citation brackets
    text = re.sub(r"\[\d+\]", "", text)

    # Normalize unicode quotes/dashes to ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Collapse excessive whitespace while preserving paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)          # multiple spaces/tabs → single space
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ newlines → paragraph break
    text = re.sub(r" \n", "\n", text)             # trailing spaces before newline

    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, target_words: int = TARGET_WORDS) -> list[str]:
    """
    Strategy:
      1. Split on paragraph boundaries (blank lines).
      2. If a paragraph is within target_words, keep it as-is.
      3. If a paragraph is too long, slide through it sentence-by-sentence
         and group sentences until the chunk reaches ~target_words.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks = []

    for para in paragraphs:
        words = para.split()
        if len(words) <= target_words:
            chunks.append(para)
        else:
            # Split long paragraph into sentence groups
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk: list[str] = []
            current_count = 0

            for sentence in sentences:
                sent_words = len(sentence.split())
                if current_count + sent_words > target_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_count = sent_words
                else:
                    current_chunk.append(sentence)
                    current_count += sent_words

            if current_chunk:
                chunks.append(" ".join(current_chunk))

    # Filter out very short fragments
    chunks = [c for c in chunks if len(c.split()) >= MIN_WORDS]
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in '{INPUT_DIR}'. Run 01_extract_text.py first.")
        return

    records = []
    passage_id = 0

    for filename in sorted(txt_files):
        source_doc = os.path.splitext(filename)[0]
        filepath   = os.path.join(INPUT_DIR, filename)

        print(f"\n📄  Processing: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = clean_text(raw)
        chunks  = split_into_chunks(cleaned)

        print(f"    → {len(raw):,} chars  |  {len(chunks)} passages after chunking")

        for chunk in chunks:
            records.append({
                "passage_id":   passage_id,
                "passage_text": chunk,
                "source_doc":   source_doc,
            })
            passage_id += 1

    df = pd.DataFrame(records, columns=["passage_id", "passage_text", "source_doc"])

    # Save in both formats
    csv_path = os.path.join(OUTPUT_DIR, "passages.csv")
    pkl_path = os.path.join(OUTPUT_DIR, "passages.pkl")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"\n✅  Done.  {len(df):,} total passages saved to:")
    print(f"   {csv_path}")
    print(f"   {pkl_path}")
    print(df.groupby("source_doc").size().rename("passage_count").to_string())


if __name__ == "__main__":
    main()