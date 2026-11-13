"""API endpoints for semantic and hybrid vector retrieval."""

import asyncio
import time
from fastapi import APIRouter, HTTPException, status

from app.core.retriever import get_retriever
from app.core.vector_store import get_vector_store
from app.models.schemas import RetrievalQuery, RetrievalResult

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])
retriever = get_retriever()
vector_store = get_vector_store()


@router.post("/search", response_model=RetrievalResult)
async def search_chunks(payload: RetrievalQuery):
    """
    Execute semantic or hybrid vector search over indexed document chunks,
    with optional cross-encoder reranking.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )

    start_time = time.perf_counter()

    try:
        chunks = await asyncio.to_thread(
            retriever.retrieve,
            query=payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
            use_reranking=payload.use_reranking,
            document_id=payload.document_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return RetrievalResult(
        query=payload.query,
        total_retrieved=len(chunks),
        retrieval_mode=payload.mode,
        reranked=payload.use_reranking,
        latency_ms=round(latency_ms, 2),
        chunks=chunks,
    )


@router.get("/stats")
async def get_vector_store_stats():
    """Retrieve FAISS index metadata and total indexed chunk count."""
    stats = vector_store.get_stats()
    return {
        "status": "online",
        "vector_store": "FAISS",
        "metric": "Inner Product (Cosine Similarity)",
        **stats,
    }
