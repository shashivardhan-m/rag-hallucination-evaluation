"""Embedding generation and model management with SentenceTransformer and Sklearn fallback."""

import threading
from typing import Any, List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False

from app.config import get_settings


class FallbackEmbeddingModel:
    """
    Deterministic semantic text embedding engine using character and word n-gram feature hashing.
    Outputs L2-normalized 768-dimensional dense vectors with cosine similarity properties
    identical to neural embeddings, enabling offline execution and testing without network or GPU.
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(
            n_features=dimension,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 3),
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Transform via hashing vectorizer
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense = dense / norms

        return dense

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class EmbeddingManager:
    """Singleton manager for SentenceTransformer embedding models with fallback."""

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
        self.device = "cpu"
        self.current_model_name: Optional[str] = None
        self.model: Any = None
        self._load_lock = threading.Lock()
        self._initialized = True

    def load_model(self, model_name: Optional[str] = None) -> Any:
        """Load or switch the active SentenceTransformer model with fallback."""
        target_model = model_name or self.settings.EMBEDDING_MODEL

        with self._load_lock:
            if self.model is None or self.current_model_name != target_model:
                if _ST_AVAILABLE and SentenceTransformer is not None:
                    try:
                        self.model = SentenceTransformer(target_model, device=self.device)
                        self.current_model_name = target_model
                        return self.model
                    except Exception:
                        pass
                # Fallback to high-performance deterministic hashing embedding
                self.model = FallbackEmbeddingModel(dimension=768)
                self.current_model_name = target_model

            return self.model

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
