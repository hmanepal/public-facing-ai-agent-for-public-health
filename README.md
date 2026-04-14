# public-facing-ai-agent-for-public-health

A RAG (Retrieval-Augmented Generation) pipeline for querying a corpus of public health documents — speeches, papers, transcripts, Wikipedia articles, and more — through a conversational AI agent.

---

## Repo Structure

```
public-facing-ai-agent-for-public-health/
├── ingest.py                 # Incremental ingest pipeline (Stages 1–3 combined)
├── 01_extract_text.py        # Stage 1 standalone: extract text from Wikipedia & PDFs
├── 02_preprocess.py          # Stage 2 standalone: clean, chunk, and structure passages
├── 03_vectorize.py           # Stage 3 standalone: generate sentence embeddings
├── 04_build_index.py         # Stage 4: validate embeddings matrix, write manifest
├── 05_rag_query.py           # Stage 5: RAG query interface (retrieval + generation)
├── requirements.txt          # Python dependencies
├── data/                     # Drop PDF files here (not committed)
├── raw_text/                 # Output of Stage 1 — raw .txt files (not committed)
└── processed/                # Output of Stages 2–4 — DataFrames, embeddings, manifest (not committed)
```

---

## Setup

**Python 3.10+ recommended.**

```bash
git clone https://github.com/<org>/public-facing-ai-agent-for-public-health.git
cd public-facing-ai-agent-for-public-health
pip install -r requirements.txt
```

If you hit torch/torchvision version conflicts, install them as a matched set first:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers --force-reinstall
pip install --upgrade peft
```

---

## Quickstart

### Adding documents and building the corpus

Drop any PDFs into `data/` and run:

```bash
python ingest.py
```

`ingest.py` combines Stages 1–3 into a single incremental pipeline. It skips any source that has already been processed and only extracts, chunks, and embeds new documents, merging them into the existing corpus. Re-run it any time you add new files.

To add Wikipedia articles, update the `WIKIPEDIA_ARTICLES` dict at the top of `ingest.py`:

```python
WIKIPEDIA_ARTICLES = {
    "wikipedia_who":     "World Health Organization",
    "wikipedia_covid19": "COVID-19 pandemic",
    # add more here...
}
```

### Validating the index

After ingesting, optionally run:

```bash
python 04_build_index.py
```

This checks that the embeddings matrix and DataFrame are aligned, normalizes vectors if needed, prints a passage breakdown by source, and writes `processed/index_manifest.json`.

### Querying

**Interactive REPL:**
```bash
python 05_rag_query.py
```

**Single query:**
```bash
python 05_rag_query.py --query "What are the WHO guidelines on vaccine distribution?"
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | None | Single query string; omit for interactive mode |
| `--top-k` | 5 | Number of passages retrieved per query |
| `--model` | `google/flan-t5-base` | HuggingFace generator model |

---

## How It Works

**Retrieval** — At query time, the question is embedded using the same SentenceTransformer model used during ingestion (`all-MiniLM-L6-v2`). Cosine similarity is computed against every passage vector via dot product (embeddings are unit-normalized), and the top-k passages are returned.

**Generation** — The retrieved passages are assembled into a prompt with source attribution and passed to Flan-T5, a free open-source instruction-tuned model that runs locally with no API key required. The model generates a grounded answer from the provided context only.

**Output** — Each response includes the generated answer and a list of retrieved passages with their source document and similarity score.

---

## DataFrame Schema

| Column | Type | Description |
|--------|------|-------------|
| `passage_id` | int | Unique passage identifier |
| `passage_text` | str | Cleaned text chunk (~150 words) |
| `source_doc` | str | Source filename (without extension) |
| `embedding` | np.ndarray | Unit-normalized embedding vector (384-dim) |

---

## Key Configuration Options

| File | Variable | Default | Notes |
|------|----------|---------|-------|
| `ingest.py` | `TARGET_WORDS` | `150` | Target passage size in words |
| `ingest.py` | `MIN_WORDS` | `20` | Passages shorter than this are discarded |
| `ingest.py` | `MODEL_NAME` | `all-MiniLM-L6-v2` | Swap for `all-mpnet-base-v2` (higher quality) or `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical domain) |
| `ingest.py` | `BATCH_SIZE` | `64` | Reduce if hitting memory issues on CPU |
| `05_rag_query.py` | `DEFAULT_TOP_K` | `5` | Default number of passages retrieved |
| `05_rag_query.py` | `GENERATOR_MODEL` | `google/flan-t5-base` | Swap for `flan-t5-large` for better quality |
| `05_rag_query.py` | `MAX_NEW_TOKENS` | `256` | Max tokens in generated answer |

---

## Dependencies

See `requirements.txt`. Key packages:

- `wikipedia-api` — Wikipedia text fetching
- `PyPDF2` — PDF text extraction
- `pandas` — passage DataFrame
- `sentence-transformers` — embedding model (`all-MiniLM-L6-v2`)
- `transformers` — Flan-T5 generator
- `torch` — backend for both models
- `sentencepiece` — Flan-T5 tokenizer

---

## Known Limitations

- **Scanned PDFs** — PyPDF2 cannot extract text from image-based pages. Affected pages are skipped with a warning. OCR support (`pytesseract`) is not yet implemented.
- **No chunking overlap** — passages are non-overlapping, which may hurt retrieval on queries that span chunk boundaries.
- **Flan-T5 answer quality** — the local model is fast and free but produces shorter, less nuanced answers than a frontier model. Swap in the Anthropic or OpenAI API for production-quality generation.
- `data/`, `raw_text/`, and `processed/` are not committed to the repo.

---

## Roadmap

- [ ] OCR support for scanned PDFs
- [ ] Chunking with sliding window overlap
- [ ] FAISS index for large-scale retrieval
- [ ] Swap Flan-T5 for API-backed LLM (Claude / GPT-4)
- [ ] Support for additional source types (HTML, `.docx`, RSS feeds)
- [ ] Web UI / API wrapper for the query interface

---

*Last updated: April 2026*