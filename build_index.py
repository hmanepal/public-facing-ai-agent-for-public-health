"""
Stage 4: Build Retrieval Index
==============================
Loads the embeddings matrix and DataFrame produced in Stage 3 and validates
that everything is in order for retrieval. For our scale, the numpy matrix
IS the index — no FAISS needed yet.

Saves a lightweight index manifest so Stage 5 knows what it's working with.

Run once after Stage 3, and again any time you add new documents.

Install deps:
    pip install numpy pandas
"""

import os
import json
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────

PROCESSED_DIR = "processed"
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "embeddings_matrix.npy")
PASSAGES_PATH   = os.path.join(PROCESSED_DIR, "passages_with_embeddings.pkl")
MANIFEST_PATH   = os.path.join(PROCESSED_DIR, "index_manifest.json")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # --- Validate inputs ---
    for path in [EMBEDDINGS_PATH, PASSAGES_PATH]:
        if not os.path.exists(path):
            print(f"✗  Missing: {path}")
            print("   Run 03_vectorize.py first.")
            return

    # --- Load ---
    print("\n📦  Loading passages and embeddings...")
    df          = pd.read_pickle(PASSAGES_PATH)
    embeddings  = np.load(EMBEDDINGS_PATH)

    print(f"    Passages : {len(df):,}")
    print(f"    Matrix   : {embeddings.shape}  (passages × embedding_dim)")

    # --- Sanity checks ---
    assert len(df) == embeddings.shape[0], (
        f"Row count mismatch: DataFrame has {len(df)} rows "
        f"but embeddings matrix has {embeddings.shape[0]}"
    )
    assert "passage_text" in df.columns and "source_doc" in df.columns, (
        "DataFrame is missing expected columns."
    )

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("  ⚠  Embeddings are not unit-normalized. Normalizing now...")
        embeddings = embeddings / norms[:, np.newaxis]
        np.save(EMBEDDINGS_PATH, embeddings)
        print("     Re-saved normalized embeddings.")

    # --- Source breakdown ---
    source_counts = df.groupby("source_doc").size().to_dict()
    print("\n    Passages by source:")
    for source, count in sorted(source_counts.items()):
        print(f"      {source:<40} {count:>5} passages")

    # --- Write manifest ---
    manifest = {
        "num_passages"   : int(len(df)),
        "embedding_dim"  : int(embeddings.shape[1]),
        "sources"        : source_counts,
        "embeddings_path": EMBEDDINGS_PATH,
        "passages_path"  : PASSAGES_PATH,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅  Index ready. Manifest saved → {MANIFEST_PATH}")
    print("    Run 05_rag_query.py to start querying.")


if __name__ == "__main__":
    main()