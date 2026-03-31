# public-facing-ai-agent-for-public-health
Public Facing AI Agent for Public Health AI Knowledge
# public-facing-ai-agent-for-public-health

A RAG (Retrieval-Augmented Generation) pipeline for querying a corpus of public health documents — speeches, papers, transcripts, Wikipedia articles, and more — through a conversational AI agent.

---

## Project Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Text extraction (Wikipedia, PDF) | ✅ Done |
| 2 | Preprocessing & chunking | ✅ Done |
| 3 | Vectorization (embeddings) | ✅ Done |
| 4 | Vector search / retrieval | 🔲 Not started |
| 5 | RAG agent / query interface | 🔲 Not started |

---

## Repo Structure

```
public-facing-ai-agent-for-public-health/
├── 01_extract_text.py        # Stage 1: Pull text from Wikipedia & arXiv PDFs
├── 02_preprocess.py          # Stage 2: Clean, chunk, and structure passages
├── 03_vectorize.py           # Stage 3: Generate sentence embeddings
├── data/                     # Place raw PDFs here 
├── raw_text/                 # Output of Stage 1 — raw .txt files (not committed)
└── processed/                # Output of Stages 2–3 — DataFrames & embeddings (not committed)
```

---

## Running the Pipeline

Run the three stages in order.

### Stage 1 — Extract Text

Fetches Wikipedia articles and extracts text from local arXiv PDFs.

- Configure which Wikipedia articles and PDFs to pull at the top of the script (`WIKIPEDIA_ARTICLES`, `ARXIV_PDFS`)
- Output: `.txt` files saved to `raw_text/`

### Stage 2 — Preprocess & Structure

Cleans the raw text and chunks it into passages of ~150 words.

- Strips citation brackets, PDF artefacts, unicode punctuation, extra whitespace
- Splits on paragraph boundaries; subdivides long paragraphs by sentence grouping
- Output: `processed/passages.csv` and `processed/passages.pkl`

DataFrame schema:

| Column | Type | Description |
|--------|------|-------------|
| `passage_id` | int | Unique passage identifier |
| `passage_text` | str | Cleaned text chunk (~150 words) |
| `source_doc` | str | Source filename (without extension) |

### Stage 3 — Vectorize

Generates a sentence embedding for each passage using `all-MiniLM-L6-v2`.


- Embeddings are L2-normalized (cosine similarity = dot product)
- Output: `processed/passages_with_embeddings.pkl` (DataFrame + embedding column) and `processed/embeddings_matrix.npy` (raw matrix for FAISS etc.)

---

## Adding New Sources

**Wikipedia articles** — add entries to `WIKIPEDIA_ARTICLES` in `01_extract_text.py`:
```python
WIKIPEDIA_ARTICLES = {
    "wikipedia_who.txt": "World Health Organization",
    "wikipedia_my_topic.txt": "Your Article Title Here",
}
```

**PDFs** (arXiv papers, reports, transcripts) — drop the PDF into `data/` and add an entry to `ARXIV_PDFS`:
```python
ARXIV_PDFS = {
    "my_paper.txt": "data/my_paper.pdf",
}
```

---

## Key Configuration Options

| Script | Variable | Default | Notes |
|--------|----------|---------|-------|
| `02_preprocess.py` | `TARGET_WORDS` | `150` | Target chunk size in words |
| `02_preprocess.py` | `MIN_WORDS` | `20` | Minimum passage length; shorter chunks are discarded |
| `03_vectorize.py` | `MODEL_NAME` | `all-MiniLM-L6-v2` | Swap for `all-mpnet-base-v2` (higher quality) or `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical domain) |
| `03_vectorize.py` | `BATCH_SIZE` | `64` | Reduce if hitting memory issues on CPU |

---

## Dependencies

See `requirements.txt`. Key packages:

- `wikipedia-api` — Wikipedia text fetching
- `PyPDF2` — PDF text extraction
- `pandas` — passage DataFrame
- `sentence-transformers` — embedding model
- `torch` — backend for sentence-transformers

---

## Notes & Known Limitations

- **Scanned PDFs** — PyPDF2 cannot extract text from image-based pages. If a PDF is scanned, you'll see a warning and that page will be skipped. OCR support (e.g. `pytesseract`) is not yet implemented.
- **Wikipedia API rate limits** — fetching many articles in quick succession may get throttled. Add a short `time.sleep()` between calls if needed.
- **No chunking overlap** — current chunking has no sliding window overlap between passages, which may hurt retrieval on queries that span chunk boundaries. Worth revisiting in Stage 4.

*Last updated: March 2026*