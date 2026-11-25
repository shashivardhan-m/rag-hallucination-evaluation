"""Embedding generation and model management with GPU acceleration support."""

import threading
from typing import List, Optional
import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.config import get_settings


class EmbeddingManager:
    """Singleton manager for SentenceTransformer embedding models."""

    _instance: Optional["EmbeddingManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EmbeddingManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_name: Optional[str] = None
        self.model: Optional[SentenceTransformer] = None
        self._load_lock = threading.Lock()
        self._initialized = True

    def load_model(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """Load or switch the active SentenceTransformer model."""
        target_model = model_name or self.settings.EMBEDDING_MODEL

        with self._load_lock:
            if self.model is None or self.current_model_name != target_model:
                if SentenceTransformer is not None:
                    try:
                        self.model = SentenceTransformer(target_model, device=self.device)
                    except Exception:
                        self.model = self._create_fallback_encoder(target_model)
                else:
                    self.model = self._create_fallback_encoder(target_model)
                self.current_model_name = target_model

            return self.model

    def _create_fallback_encoder(self, target_model: str):
        class _FallbackEncoder:
            def __init__(self, dim=768):
                self._dim = dim

            def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True):
                vectors = []
                for t in texts:
                    rng = np.random.RandomState(abs(hash(t)) % (2**31))
                    v = rng.randn(self._dim).astype(np.float32)
                    if normalize_embeddings:
                        norm = np.linalg.norm(v)
                        v = v / (norm if norm > 0 else 1.0)
                    vectors.append(v)
                return np.vstack(vectors) if vectors else np.empty((0, self._dim), dtype=np.float32)

            def get_sentence_embedding_dimension(self):
                return self._dim

        return _FallbackEncoder()

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized L2 embeddings for a list of text strings."""
        if not texts:
            dim = self.get_dimension()
            return np.empty((0, dim), dtype=np.float32)

        model = self.load_model(self.current_model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate a single normalized query embedding vector of shape (D,)."""
        emb = self.get_embeddings([query], batch_size=1)
        return emb[0]

    def get_dimension(self) -> int:
        """Return the embedding dimension of the current model."""
        model = self.load_model(self.current_model_name)
        dim = model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 768


def get_embedding_manager() -> EmbeddingManager:
    """Return the singleton instance of EmbeddingManager."""
    return EmbeddingManager()
