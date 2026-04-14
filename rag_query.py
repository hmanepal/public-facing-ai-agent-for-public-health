"""
Stage 5: RAG Query Interface
=============================
Retrieval-Augmented Generation over the public health corpus.

Retrieval  : Dense cosine similarity search over SentenceTransformer embeddings
Generation : google/flan-t5-base (local, free, no API key required)

Usage:
    # Interactive mode (REPL)
    python 05_rag_query.py

    # Single query via CLI
    python 05_rag_query.py --query "What are the WHO guidelines on vaccine distribution?"

    # More retrieved passages, larger model
    python 05_rag_query.py --top-k 7 --model google/flan-t5-large

Install deps:
    pip install numpy pandas sentence-transformers transformers torch sentencepiece
"""

import os
import json
import argparse
import textwrap
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Config ───────────────────────────────────────────────────────────────────

PROCESSED_DIR   = "processed"
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "embeddings_matrix.npy")
PASSAGES_PATH   = os.path.join(PROCESSED_DIR, "passages_with_embeddings.pkl")

# Must match the model used in Stage 3
RETRIEVER_MODEL = "all-MiniLM-L6-v2"

# Generator — swap for "google/flan-t5-large" for better quality (slower, more RAM)
GENERATOR_MODEL = "google/flan-t5-base"

DEFAULT_TOP_K   = 5      # number of passages to retrieve
MAX_NEW_TOKENS  = 256    # max tokens in the generated answer

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, embeddings: np.ndarray, df: pd.DataFrame,
             retriever: SentenceTransformer, top_k: int) -> pd.DataFrame:
    """
    Embed the query and return the top-k most similar passages.
    Cosine similarity = dot product (embeddings are unit-normalized).
    """
    query_vec = retriever.encode(query, normalize_embeddings=True)   # shape: (dim,)
    scores    = embeddings @ query_vec                                # shape: (n_passages,)
    top_idx   = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results


# ── Generation ────────────────────────────────────────────────────────────────

def build_prompt(query: str, passages: pd.DataFrame) -> str:
    """
    Construct the RAG prompt: context passages + question.
    Flan-T5 works well with explicit instruction framing.
    """
    context_blocks = []
    for _, row in passages.iterrows():
        context_blocks.append(
            f"[Source: {row['source_doc']}]\n{row['passage_text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a public health assistant. "
        "Answer the question using only the information in the context below. "
        "If the context does not contain enough information to answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt


def generate_answer(prompt: str, tokenizer: T5Tokenizer,
                    model: T5ForConditionalGeneration) -> str:
    """Tokenize the prompt and generate an answer with Flan-T5."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,    # flan-t5-base context window
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=4,             # beam search for better quality
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(query: str, answer: str, passages: pd.DataFrame) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  QUERY : {query}")
    print("─" * width)
    print(f"  ANSWER\n")
    for line in textwrap.wrap(answer, width - 2):
        print(f"  {line}")
    print("\n" + "─" * width)
    print(f"  RETRIEVED PASSAGES  (top {len(passages)})\n")
    for i, (_, row) in enumerate(passages.iterrows(), 1):
        print(f"  [{i}] {row['source_doc']}  (score: {row['score']:.3f})")
        snippet = row["passage_text"][:200].replace("\n", " ")
        print(f"      {snippet}...")
        print()
    print("═" * width)


# ── Main ─────────────────────────────────────────────────────────────────────

def load_resources(generator_model: str):
    """Load all models and data once, then reuse across queries."""

    for path in [EMBEDDINGS_PATH, PASSAGES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\nRun 03_vectorize.py and 04_build_index.py first."
            )

    print("\n📦  Loading passages and embeddings...")
    df         = pd.read_pickle(PASSAGES_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"    {len(df):,} passages ready.")

    print(f"\n🔍  Loading retriever: {RETRIEVER_MODEL}")
    retriever = SentenceTransformer(RETRIEVER_MODEL)

    print(f"\n🤖  Loading generator: {generator_model}  (first run downloads the model)")
    tokenizer = T5Tokenizer.from_pretrained(generator_model)
    model     = T5ForConditionalGeneration.from_pretrained(generator_model)
    model.eval()
    print("    Ready.\n")

    return df, embeddings, retriever, tokenizer, model


def run_query(query: str, df, embeddings, retriever, tokenizer, model, top_k: int):
    passages = retrieve(query, embeddings, df, retriever, top_k)
    prompt   = build_prompt(query, passages)
    answer   = generate_answer(prompt, tokenizer, model)
    print_results(query, answer, passages)


def main():
    parser = argparse.ArgumentParser(description="RAG query interface for public health corpus")
    parser.add_argument("--query",   type=str,  default=None,           help="Single query string")
    parser.add_argument("--top-k",   type=int,  default=DEFAULT_TOP_K,  help="Number of passages to retrieve")
    parser.add_argument("--model",   type=str,  default=GENERATOR_MODEL,help="HuggingFace generator model name")
    args = parser.parse_args()

    df, embeddings, retriever, tokenizer, model = load_resources(args.model)

    if args.query:
        # Single query mode
        run_query(args.query, df, embeddings, retriever, tokenizer, model, args.top_k)
    else:
        # Interactive REPL
        print("💬  RAG Query Interface  (type 'exit' to quit)\n")
        while True:
            try:
                query = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                break
            run_query(query, df, embeddings, retriever, tokenizer, model, args.top_k)


if __name__ == "__main__":
    main()