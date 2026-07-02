"""
Minimal in-memory vector store — replaces ChromaDB.

Why: ChromaDB relies on native SQLite bindings that can be flaky on some
cloud hosting environments (observed AttributeError crashes tied to its
internal system-client teardown on Streamlit Community Cloud). Since we
already compute embeddings ourselves via sentence-transformers, a plain
numpy cosine-similarity store does the same job for this app's scale
(a handful of uploaded documents) with far fewer moving parts.

Not persisted to disk — cleared when the Streamlit session ends, same
behavior as the previous EphemeralClient setup.
"""
import numpy as np


class VectorStore:
    def __init__(self):
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def count(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        self._ids.extend(ids)
        self._vectors.extend(embeddings)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

    def query(self, query_embedding: list[float], n_results: int = 5) -> dict:
        if not self._vectors:
            return {"documents": [[]], "metadatas": [[]]}

        matrix = np.array(self._vectors)
        query_vec = np.array(query_embedding)

        # cosine similarity
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        similarities = (matrix @ query_vec) / (matrix_norms * query_norm + 1e-10)

        top_idx = np.argsort(similarities)[::-1][:n_results]
        docs = [self._documents[i] for i in top_idx]
        metas = [self._metadatas[i] for i in top_idx]
        return {"documents": [docs], "metadatas": [metas]}

    def get(self, where: dict) -> dict:
        """Filter by metadata, e.g. {'source': 'file.pdf'}."""
        key, value = next(iter(where.items()))
        docs = [d for d, m in zip(self._documents, self._metadatas) if m.get(key) == value]
        return {"documents": docs}
