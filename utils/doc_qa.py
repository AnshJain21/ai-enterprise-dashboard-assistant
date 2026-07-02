"""
Retrieval-augmented Q&A / summarization over uploaded documents.

Flow:
  1. Chunk each document.
  2. Embed chunks with Gemini's embedding model, store in a local
     in-memory ChromaDB collection (per Streamlit session).
  3. On a question, embed the question, retrieve the top-k nearest
     chunks, and ask the model to answer using only those chunks.
"""
import uuid
import chromadb
from utils.llm_client import chat, embed
from utils.doc_loader import extract_text, chunk_text


def get_collection():
    """Fresh in-memory Chroma client per Streamlit session (see app.py)."""
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name="documents")


def add_document(collection, filename: str, file_bytes: bytes) -> int:
    """Extracts, chunks, embeds, and stores one document. Returns #chunks added."""
    text = extract_text(filename, file_bytes)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = embed(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
    return len(chunks)


def ask(collection, question: str, top_k: int = 5) -> dict:
    """Retrieve relevant chunks and answer the question grounded in them."""
    if collection.count() == 0:
        return {"answer": "No documents have been uploaded yet.", "sources": []}

    q_vector = embed([question])[0]
    results = collection.query(query_embeddings=[q_vector], n_results=min(top_k, collection.count()))

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n---\n\n".join(docs)

    system = (
        "Answer the user's question using ONLY the provided document excerpts. "
        "If the excerpts don't contain the answer, say so clearly instead of guessing."
    )
    prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}"
    answer = chat(prompt, system_instruction=system)

    sources = sorted({m["source"] for m in metas})
    return {"answer": answer, "sources": sources}


def summarize_document(collection, filename: str) -> str:
    """Summarize one previously-added document by pulling all its chunks."""
    results = collection.get(where={"source": filename})
    docs = results["documents"]
    if not docs:
        return f"No content found for {filename}."

    full_text = "\n\n".join(docs)
    prompt = (
        f"Summarize the following document in a concise executive summary "
        f"(bullet points for key facts, 1 short paragraph for context):\n\n{full_text}"
    )
    return chat(prompt)
