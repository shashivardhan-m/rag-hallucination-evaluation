"""Cross-encoder reranking for refining document retrieval candidate ordering."""

import math
import threading
from typing import Any, Dict, List, Optional
import torch
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from app.config import get_settings


class Reranker:
    """Reranker using a CrossEncoder transformer model."""

    _instance: Optional["Reranker"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "Reranker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Reranker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = self.settings.RERANKER_MODEL
        self.model: Optional[CrossEncoder] = None
        self._load_lock = threading.Lock()
        self._initialized = True

    def _load_model(self):
        """Lazy load the CrossEncoder model onto CUDA/CPU."""
        with self._load_lock:
            if self.model is None:
                if CrossEncoder is not None:
                    try:
                        self.model = CrossEncoder(self.model_name, device=self.device)
                    except Exception:
                        self.model = self._create_fallback_reranker()
                else:
                    self.model = self._create_fallback_reranker()
            return self.model

    def _create_fallback_reranker(self):
        class _FallbackReranker:
            def predict(self, pairs):
                scores = []
                for q, d in pairs:
                    q_words = set(q.lower().split())
                    d_words = set(d.lower().split())
                    jaccard = len(q_words & d_words) / max(1, len(q_words | d_words))
                    scores.append(float(jaccard * 4.0 - 1.0))
                return np.array(scores)
        return _FallbackReranker()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Convert cross-encoder logit into normalized [0, 1] probability."""
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunk dictionaries against the user query.
        Returns top_n chunks sorted in descending order of reranker relevance score.
        """
        if not chunks:
            return []

        if len(chunks) == 1:
            chunk = dict(chunks[0])
            chunk["rank"] = 1
            return [chunk]

        model = self._load_model()
        pairs = [[query, chunk.get("content", "")] for chunk in chunks]

        scores = model.predict(pairs)

        reranked_chunks: List[Dict[str, Any]] = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            norm_score = self._sigmoid(float(score))
            item["score"] = round(norm_score, 4)
            reranked_chunks.append(item)

        # Sort descending by score
        reranked_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Re-assign ranks
        for idx, item in enumerate(reranked_chunks[:top_n], start=1):
            item["rank"] = idx

        return reranked_chunks[:top_n]


def get_reranker() -> Reranker:
    """Return singleton Reranker instance."""
    return Reranker()
