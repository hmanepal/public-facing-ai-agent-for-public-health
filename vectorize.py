"""
Stage 3: Vectorization
=======================
Loads `processed/passages.pkl`, generates an embedding vector for each passage
using a SentenceTransformer model, and saves the enriched DataFrame back to disk.

New column added to the DataFrame:
    embedding  – np.ndarray of shape (embedding_dim,)

Install deps:
    pip install sentence-transformers torch pandas numpy
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = "processed"
INPUT_PKL     = os.path.join(PROCESSED_DIR, "passages.pkl")

# Output files
OUTPUT_PKL    = os.path.join(PROCESSED_DIR, "passages_with_embeddings.pkl")
OUTPUT_NPY    = os.path.join(PROCESSED_DIR, "embeddings_matrix.npy")

# Model choices (all free, run locally):
#   "all-MiniLM-L6-v2"          – fast, good quality, 384-dim  (recommended default)
#   "all-mpnet-base-v2"         – slower, higher quality, 768-dim
#   "pritamdeka/S-PubMedBert-MS-MARCO"  – biomedical domain, good for health text
MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64   # lower if you hit OOM on CPU

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load passages
    if not os.path.exists(INPUT_PKL):
        print(f"Input file not found: {INPUT_PKL}")
        print("Run 02_preprocess.py first.")
        return

    df = pd.read_pickle(INPUT_PKL)
    print(f"\n📦  Loaded {len(df):,} passages from {INPUT_PKL}")

    # Load model
    print(f"\n🤖  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"    Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings
    print(f"\n⚙️   Encoding passages (batch_size={BATCH_SIZE})...")
    passages = df["passage_text"].tolist()

    embeddings = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-normalize → cosine sim = dot product
    )

    print(f"\n    Embedding matrix shape: {embeddings.shape}")

    # Store embeddings in DataFrame (each row is a numpy array)
    df["embedding"] = list(embeddings)

    # Save enriched DataFrame
    df.to_pickle(OUTPUT_PKL)
    print(f"\n✅  Saved enriched DataFrame → {OUTPUT_PKL}")

    # Also save the raw matrix as .npy for fast loading into FAISS / sklearn later
    np.save(OUTPUT_NPY, embeddings)
    print(f"    Saved embeddings matrix  → {OUTPUT_NPY}")

    # Quick sanity check
    print("\n📊  Sample:")
    print(df[["passage_id", "source_doc", "passage_text"]].head(3).to_string(index=False))
    print(f"\n    Embedding for passage 0: shape={df['embedding'].iloc[0].shape}, "
          f"norm={np.linalg.norm(df['embedding'].iloc[0]):.4f}")


if __name__ == "__main__":
    main()