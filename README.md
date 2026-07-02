# AI Enterprise Dashboard Assistant

A Streamlit app that answers questions about your data (CSV/Excel) and
summarizes/answers questions about your documents (PDF/DOCX/TXT) —
built entirely on **free** tiers, no OpenAI subscription required.

## Stack
- **LLM (chat/summarization):** Groq API (free tier — no credit card needed)
- **Embeddings:** local, via `sentence-transformers` (no API call, no cost)
- **UI:** Streamlit
- **Vector store:** In-memory numpy vector store (no external DB needed)
- **Data handling:** pandas

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Get a free key at https://console.groq.com/keys and paste it into `.env`:

```
GROQ_API_KEY=your_key_here
```

Then run:

```bash
streamlit run app.py
```

## How it works

### Data Q&A tab
1. Upload a CSV/Excel file.
2. Ask a question in plain English.
3. The model writes a single pandas expression against your DataFrame,
   the app executes it locally (in a restricted sandbox — no file I/O,
   `eval`, or imports allowed), and the model explains the *actual
   computed result* in plain language. This keeps every number grounded
   in your real data instead of the model guessing.

### Document Summarization tab
1. Upload one or more PDF/DOCX/TXT files.
2. Each document is chunked, embedded, and stored in a local in-memory vector store (see utils/vector_store.py)
   collection (cleared when the session ends — nothing persists to disk).
3. "Summarize" pulls all chunks for one document and asks the model for
   an executive summary.
4. "Ask questions across all documents" retrieves the most relevant
   chunks for your question (RAG) and answers using only those excerpts,
   citing which source file(s) it used.

## Swapping the LLM provider later
All model calls go through `utils/llm_client.py`. To switch to OpenRouter,
a local Ollama model, or back to a cloud provider, you only need to edit
that one file — `data_qa.py` and `doc_qa.py` don't need to change.

## Notes on the free tier
Groq's free tier has rate limits (requests per minute/day) that are fine
for internal/team use but not for a public-facing high-traffic app. If you
outgrow it, OpenRouter (aggregates many providers) or a local Ollama model
are good next options — see `utils/llm_client.py` for where to swap it in.
