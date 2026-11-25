"""FAISS vector store for semantic similarity search over document chunks with NumPy fallback."""

import os
import pickle
import threading
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    _FAISS_AVAILABLE = False

from app.config import get_settings


class FAISSVectorStore:
    """In-memory and persisted vector index using Inner Product (cosine similarity)."""

    _instance: Optional["FAISSVectorStore"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FAISSVectorStore":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(FAISSVectorStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.lock = threading.RLock()
        self.dimension: Optional[int] = None
        self.index: Any = None
        self.metadata: List[Dict[str, Any]] = []
        self.all_embeddings: Optional[np.ndarray] = None
        self._initialized = True

    def _init_index(self, dimension: int) -> None:
        """Initialize empty IndexFlatIP for the given embedding dimension."""
        self.dimension = dimension
        if _FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # Native NumPy fallback index (all_embeddings will store the vectors)
            self.index = "numpy"

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 normalize row vectors."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        document_id: str,
    ) -> None:
        """Add chunks and corresponding normalized embeddings to the index."""
        if not chunks or embeddings.size == 0:
            return

        with self.lock:
            embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
            n_samples, dim = embeddings_f32.shape

            if self.index is None:
                self._init_index(dim)
            elif dim != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {dim}"
                )

            # Ensure vectors are L2-normalized for cosine similarity
            if _FAISS_AVAILABLE and faiss is not None:
                faiss.normalize_L2(embeddings_f32)
                self.index.add(embeddings_f32)
            else:
                embeddings_f32 = self._normalize(embeddings_f32)

            # Keep all embeddings for rebuilds on deletion and NumPy search
            if self.all_embeddings is None or self.all_embeddings.size == 0:
                self.all_embeddings = embeddings_f32.copy()
            else:
                self.all_embeddings = np.vstack([self.all_embeddings, embeddings_f32])

            for chunk in chunks:
                chunk_copy = dict(chunk)
                chunk_copy.setdefault("document_id", document_id)
                self.metadata.append(chunk_copy)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search top-k most similar chunks for query embedding."""
        with self.lock:
            total_items = len(self.metadata)
            if total_items == 0 or self.all_embeddings is None or self.all_embeddings.size == 0:
                return []

            q = np.ascontiguousarray(query_embedding, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)

            if _FAISS_AVAILABLE and faiss is not None and self.index is not None and not isinstance(self.index, str):
                faiss.normalize_L2(q)
                k_search = min(self.index.ntotal, top_k * 4 if document_id else top_k)
                distances, indices = self.index.search(q, k_search)
                raw_indices = indices[0]
                raw_scores = distances[0]
            else:
                # NumPy cosine similarity search
                q_norm = self._normalize(q)
                sims = np.dot(self.all_embeddings, q_norm.T).flatten()
                k_search = min(len(sims), top_k * 4 if document_id else top_k)
                raw_indices = np.argsort(-sims)[:k_search]
                raw_scores = sims[raw_indices]

            results: List[Dict[str, Any]] = []
            for idx, score in zip(raw_indices, raw_scores):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                if document_id and meta.get("document_id") != document_id:
                    continue

                res = dict(meta)
                res["score"] = float(score)
                results.append(res)
                if len(results) >= top_k:
                    break

            return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return all metadata chunks stored in the index."""
        with self.lock:
            return [dict(m) for m in self.metadata]

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Return metadata for chunks belonging to a specific document."""
        with self.lock:
            return [dict(m) for m in self.metadata if m.get("document_id") == document_id]

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks associated with a document and rebuild index."""
        with self.lock:
            indices_to_keep = [
                i for i, m in enumerate(self.metadata)
                if m.get("document_id") != document_id
            ]
            deleted_count = len(self.metadata) - len(indices_to_keep)
            if deleted_count == 0:
                return 0

            self.metadata = [self.metadata[i] for i in indices_to_keep]

            if self.all_embeddings is not None and len(indices_to_keep) > 0:
                self.all_embeddings = self.all_embeddings[indices_to_keep]
                if _FAISS_AVAILABLE and faiss is not None and self.dimension:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(self.all_embeddings)
            else:
                self.all_embeddings = None
                self.index = None

            return deleted_count

    def clear(self) -> None:
        """Completely reset the vector index and metadata."""
        with self.lock:
            self.index = None
            self.dimension = None
            self.metadata = []
            self.all_embeddings = None

    def get_stats(self) -> Dict[str, Any]:
        """Return current status and size of the vector store."""
        with self.lock:
            ntotal = len(self.metadata)
            return {
                "total_chunks": ntotal,
                "dimension": self.dimension or 0,
                "is_empty": ntotal == 0,
                "backend": "faiss" if _FAISS_AVAILABLE else "numpy_fallback",
            }


def get_vector_store() -> FAISSVectorStore:
    """Return singleton vector store instance."""
    return FAISSVectorStore()
