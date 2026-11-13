"""FAISS vector store for semantic similarity search over document chunks."""

import os
import pickle
import threading
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from app.config import get_settings


class FAISSVectorStore:
    """In-memory and persisted FAISS vector index using Inner Product (cosine similarity)."""

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
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.all_embeddings: Optional[np.ndarray] = None
        self._initialized = True

    def _init_index(self, dimension: int) -> None:
        """Initialize empty IndexFlatIP for the given embedding dimension."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

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
            faiss.normalize_L2(embeddings_f32)

            self.index.add(embeddings_f32)

            # Keep all embeddings for rebuilds on deletion
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
            if self.index is None or self.index.ntotal == 0 or len(self.metadata) == 0:
                return []

            q = np.ascontiguousarray(query_embedding, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            faiss.normalize_L2(q)

            k_search = min(self.index.ntotal, top_k * 4 if document_id else top_k)
            distances, indices = self.index.search(q, k_search)

            results: List[Dict[str, Any]] = []
            for rank_idx, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                item = dict(self.metadata[idx])
                if document_id and item.get("document_id") != document_id:
                    continue

                item["score"] = float(score)
                item["rank"] = len(results) + 1
                results.append(item)

                if len(results) >= top_k:
                    break

            return results

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks associated with a document and rebuild index."""
        with self.lock:
            if not self.metadata:
                return 0

            keep_indices: List[int] = []
            deleted_count = 0
            for idx, meta in enumerate(self.metadata):
                if meta.get("document_id") == document_id:
                    deleted_count += 1
                else:
                    keep_indices.append(idx)

            if deleted_count == 0:
                return 0

            new_metadata = [self.metadata[i] for i in keep_indices]
            if keep_indices and self.all_embeddings is not None and self.dimension is not None:
                new_embeddings = self.all_embeddings[keep_indices]
                new_index = faiss.IndexFlatIP(self.dimension)
                new_index.add(np.ascontiguousarray(new_embeddings, dtype=np.float32))

                self.metadata = new_metadata
                self.all_embeddings = new_embeddings
                self.index = new_index
            else:
                self.metadata = []
                self.all_embeddings = None
                if self.dimension:
                    self.index = faiss.IndexFlatIP(self.dimension)

            return deleted_count

    def save_index(self, path: Optional[str] = None) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        target_path = path or self.settings.FAISS_INDEX_PATH
        os.makedirs(target_path, exist_ok=True)

        with self.lock:
            index_file = os.path.join(target_path, "faiss.index")
            meta_file = os.path.join(target_path, "metadata.pkl")

            if self.index is not None:
                faiss.write_index(self.index, index_file)

            payload = {
                "metadata": self.metadata,
                "dimension": self.dimension,
                "all_embeddings": self.all_embeddings,
            }
            with open(meta_file, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load FAISS index and chunk metadata from disk if available."""
        target_path = path or self.settings.FAISS_INDEX_PATH
        index_file = os.path.join(target_path, "faiss.index")
        meta_file = os.path.join(target_path, "metadata.pkl")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            return False

        with self.lock:
            try:
                self.index = faiss.read_index(index_file)
                with open(meta_file, "rb") as f:
                    payload = pickle.load(f)

                self.metadata = payload.get("metadata", [])
                self.dimension = payload.get("dimension", self.index.d if self.index else None)
                self.all_embeddings = payload.get("all_embeddings", None)
                return True
            except Exception:
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics on the indexed vectors and documents."""
        with self.lock:
            total_chunks = self.index.ntotal if self.index else 0
            doc_ids = {m.get("document_id") for m in self.metadata if m.get("document_id")}
            return {
                "total_chunks": total_chunks,
                "total_documents": len(doc_ids),
                "dimension": self.dimension or 0,
            }

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return shallow copies of all stored chunk metadata."""
        with self.lock:
            return [dict(m) for m in self.metadata]


def get_vector_store() -> FAISSVectorStore:
    """Return singleton FAISSVectorStore instance."""
    return FAISSVectorStore()
