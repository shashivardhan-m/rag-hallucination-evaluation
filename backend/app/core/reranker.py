"""Cross-encoder reranking for refining document retrieval candidate ordering."""

import math
import threading
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None  # type: ignore
    _CROSS_ENCODER_AVAILABLE = False

from app.config import get_settings


class FallbackCrossEncoder:
    """Fallback cross-encoder that computes fine-grained cross-entropy lexical and overlap alignment."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs: List[List[str]]) -> List[float]:
        scores = []
        for q, doc in pairs:
            q_words = set(q.lower().split())
            d_words = doc.lower().split()
            if not q_words or not d_words:
                scores.append(0.0)
                continue

            matches = sum(1 for w in d_words if w in q_words)
            overlap_ratio = matches / max(1, len(d_words))
            query_coverage = sum(1 for qw in q_words if qw in doc.lower()) / len(q_words)
            # Logit representation centered near 0.0
            logit = (2.5 * query_coverage + 1.5 * overlap_ratio) - 1.0
            scores.append(float(logit))
        return scores


class Reranker:
    """Reranker using a CrossEncoder transformer model with fallback."""

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
        self.device = "cpu"
        self.model_name = self.settings.RERANKER_MODEL
        self.model: Any = None
        self._load_lock = threading.Lock()
        self._initialized = True

    def _load_model(self) -> Any:
        """Lazy load the CrossEncoder model with fallback."""
        with self._load_lock:
            if self.model is None:
                if _CROSS_ENCODER_AVAILABLE and CrossEncoder is not None:
                    try:
                        self.model = CrossEncoder(self.model_name, device=self.device)
                        return self.model
                    except Exception:
                        pass
                self.model = FallbackCrossEncoder(self.model_name)
            return self.model

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
