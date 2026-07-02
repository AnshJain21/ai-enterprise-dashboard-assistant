"""
Thin wrapper around the Groq API (OpenAI-compatible) so the rest of the
app never touches the SDK/HTTP details directly. Swap this file out if
you later switch providers again — everything else stays the same.

Why Groq instead of Gemini: as of early July 2026 there is an active,
widely-reported server-side bug on Google's end affecting newly-issued
"AQ."-format Gemini API keys, causing every generateContent call to fail
with a 401 regardless of configuration. Groq's free tier is simple
Bearer-token auth and has no such issue.
"""
import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env "
            "and add your free key from https://console.groq.com/keys"
        )
    return api_key


def chat(prompt: str, system_instruction: str | None = None, model: str | None = None) -> str:
    """Single-turn text generation. Returns the model's plain text reply."""
    model = model or os.environ.get("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {_get_api_key()}"},
        json={"model": model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embeddings run locally — see utils/local_embeddings.py."""
    from utils.local_embeddings import embed as local_embed
    return local_embed(texts)
