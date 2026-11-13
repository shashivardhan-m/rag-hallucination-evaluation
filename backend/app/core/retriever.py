"""Document chunk retrieval combining semantic vector search, BM25 hybrid search, and reranking."""

import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.embeddings import EmbeddingManager, get_embedding_manager
from app.core.reranker import Reranker, get_reranker
from app.core.vector_store import FAISSVectorStore, get_vector_store
from app.models.schemas import RetrievedChunk
from app.utils.text_utils import clean_text


class Retriever:
    """Retrieves relevant document chunks using dense semantic and BM25 sparse hybrid ranking."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.reranker = reranker or get_reranker()

    def _to_retrieved_chunk(self, item: Dict[str, Any], rank: int) -> RetrievedChunk:
        """Convert a raw dictionary to a validated RetrievedChunk schema."""
        chunk_id = str(item.get("id") or item.get("chunk_id") or uuid.uuid4())
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=str(item.get("document_id", "")),
            filename=str(item.get("filename", "Unknown")),
            chunk_index=int(item.get("chunk_index", 0)),
            content=str(item.get("content", "")),
            score=round(float(item.get("score", 0.0)), 4),
            rank=rank,
        )

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top_k chunks using pure dense vector inner-product search."""
        if not query.strip():
            return []

        query_emb = self.embedding_manager.get_query_embedding(query)
        raw_results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k,
            document_id=document_id,
        )

        return [
            self._to_retrieved_chunk(item, rank=idx + 1)
            for idx, item in enumerate(raw_results)
        ]

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        document_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Combine dense semantic similarity and BM25 lexical search scores."""
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            return []

        if document_id:
            all_chunks = [c for c in all_chunks if c.get("document_id") == document_id]
            if not all_chunks:
                return []

        # 1. Dense Semantic Scores for all chunks
        query_emb = self.embedding_manager.get_query_embedding(query)
        semantic_results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=len(all_chunks),
            document_id=document_id,
        )
        semantic_scores: Dict[str, float] = {}
        for r in semantic_results:
            key = f"{r.get('document_id')}_{r.get('chunk_index')}"
            semantic_scores[key] = max(0.0, float(r.get("score", 0.0)))

        # 2. BM25 Lexical Scores
        tokenized_corpus = [
            clean_text(c.get("content", "")).lower().split()
            for c in all_chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = clean_text(query).lower().split()
        bm25_raw_scores = bm25.get_scores(query_tokens)

        # Min-max scale BM25 scores to [0, 1]
        min_b = float(np.min(bm25_raw_scores)) if len(bm25_raw_scores) > 0 else 0.0
        max_b = float(np.max(bm25_raw_scores)) if len(bm25_raw_scores) > 0 else 0.0
        range_b = max_b - min_b

        hybrid_scored: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(all_chunks):
            key = f"{chunk.get('document_id')}_{chunk.get('chunk_index')}"
            sem_score = semantic_scores.get(key, 0.0)

            if range_b > 1e-6:
                bm25_norm = (bm25_raw_scores[idx] - min_b) / range_b
            else:
                bm25_norm = 1.0 if max_b > 0 else 0.0

            combined = (1.0 - bm25_weight) * sem_score + (bm25_weight * bm25_norm)
            item = dict(chunk)
            item["score"] = float(combined)
            hybrid_scored.append(item)

        hybrid_scored.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = hybrid_scored[:top_k]

        return [
            self._to_retrieved_chunk(item, rank=idx + 1)
            for idx, item in enumerate(top_candidates)
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "semantic",
        use_reranking: bool = False,
        document_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Main retrieval interface supporting 'semantic' or 'hybrid' mode with optional cross-encoder reranking.
        """
        # Retrieve extra candidates if reranking is enabled
        fetch_k = top_k * 3 if use_reranking else top_k

        if mode.lower() == "hybrid":
            candidates = self.retrieve_hybrid(query, top_k=fetch_k, document_id=document_id)
        else:
            candidates = self.retrieve_semantic(query, top_k=fetch_k, document_id=document_id)

        if not candidates:
            return []

        if not use_reranking or self.reranker is None:
            return candidates[:top_k]

        # Convert to dictionary format for CrossEncoder reranking
        raw_chunks = [c.model_dump() for c in candidates]
        reranked_raw = self.reranker.rerank(query=query, chunks=raw_chunks, top_n=top_k)

        return [
            self._to_retrieved_chunk(item, rank=idx + 1)
            for idx, item in enumerate(reranked_raw)
        ]


def get_retriever() -> Retriever:
    """Return an instantiated Retriever."""
    return Retriever()
