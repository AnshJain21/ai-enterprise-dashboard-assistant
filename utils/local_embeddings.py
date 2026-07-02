"""
Local, offline embeddings using sentence-transformers.

Why local instead of the Gemini embedding API: Google's embed_content
endpoint has a known intermittent authentication bug (unrelated to key
validity) that unpredictably rejects otherwise-valid API keys. Running
embeddings locally sidesteps that entirely, is free, has no rate limit,
and is actually faster (no network round-trip per chunk).

Chat/summarization still uses the real Gemini API (utils/llm_client.chat) —
only the embedding step runs locally.
"""
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    """Lazily load the model once and reuse it (loading takes a few seconds)."""
    global _model
    if _model is None:
        # all-MiniLM-L6-v2: small (~80MB), fast on CPU, good enough quality
        # for retrieval over reports/documents. Downloads once, then caches
        # locally under ~/.cache/torch/sentence_transformers/
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Returns one vector per input string."""
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()
