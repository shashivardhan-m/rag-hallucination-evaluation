"""API endpoints for question-answering with standard LLM, RAG pipeline, and session history."""

import asyncio
import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.rag_pipeline import get_rag_pipeline
from app.database import get_db
from app.models.db_models import DocumentChunk, QASession
from app.models.schemas import (
    AnswerResult,
    QARequest,
    QAResponse,
    QASessionInfo,
    QASessionListResponse,
)

router = APIRouter(prefix="/qa", tags=["Question Answering"])
settings = get_settings()
rag_pipeline = get_rag_pipeline()


@router.post("/ask", response_model=QAResponse)
async def ask_question(payload: QARequest, db: Session = Depends(get_db)):
    """
    Execute side-by-side comparison answering: generates both standard LLM answer
    and grounded RAG answer, recording performance metrics into a persistent QASession.
    """
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    model_config = {
        "backend": payload.backend or settings.DEFAULT_LLM_BACKEND,
        "model": payload.model,
    }

    standard_res: Optional[AnswerResult] = None
    rag_res: Optional[AnswerResult] = None
    docs_used = []

    # Run Standard Answer
    if payload.run_standard:
        try:
            standard_res = await asyncio.to_thread(
                rag_pipeline.answer_standard,
                question=payload.question,
                model_config=model_config,
            )
        except Exception as e:
            standard_res = AnswerResult(
                answer=f"[Error generating standard answer: {str(e)}]",
                latency_ms=0.0,
                model=payload.model or "unknown",
                backend=payload.backend or settings.DEFAULT_LLM_BACKEND,
            )

    # Run RAG Answer
    if payload.run_rag:
        try:
            rag_res = await asyncio.to_thread(
                rag_pipeline.answer_with_rag,
                question=payload.question,
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                use_reranking=payload.use_reranking,
                model_config=model_config,
                document_id=payload.document_id,
            )
            if rag_res.retrieved_chunks:
                docs_used = list({c.filename for c in rag_res.retrieved_chunks if c.filename})
        except Exception as e:
            rag_res = AnswerResult(
                answer=f"[Error generating RAG answer: {str(e)}]",
                latency_ms=0.0,
                model=payload.model or "unknown",
                backend=payload.backend or settings.DEFAULT_LLM_BACKEND,
            )

    # Persist session to database
    session_id = str(uuid.uuid4())
    model_name = (rag_res.model if rag_res else None) or (standard_res.model if standard_res else "default")

    db_session = QASession(
        id=session_id,
        question=payload.question,
        standard_answer=standard_res.answer if standard_res else None,
        rag_answer=rag_res.answer if rag_res else None,
        standard_latency_ms=standard_res.latency_ms if standard_res else 0.0,
        rag_latency_ms=rag_res.latency_ms if rag_res else 0.0,
        top_k=payload.top_k,
        model_used=model_name,
        embedding_model=settings.EMBEDDING_MODEL,
        documents_used=json.dumps(docs_used),
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return QAResponse(
        session_id=db_session.id,
        question=db_session.question,
        standard_answer=standard_res,
        rag_answer=rag_res,
        created_at=db_session.session_time,
    )


@router.post("/ask/standard", response_model=AnswerResult)
async def ask_standard_only(payload: QARequest, db: Session = Depends(get_db)):
    """Answer a question directly using general LLM knowledge (no retrieval)."""
    if not payload.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    model_config = {
        "backend": payload.backend or settings.DEFAULT_LLM_BACKEND,
        "model": payload.model,
    }

    res = await asyncio.to_thread(
        rag_pipeline.answer_standard,
        question=payload.question,
        model_config=model_config,
    )

    # Save session
    db_session = QASession(
        id=str(uuid.uuid4()),
        question=payload.question,
        standard_answer=res.answer,
        rag_answer=None,
        standard_latency_ms=res.latency_ms,
        rag_latency_ms=0.0,
        top_k=0,
        model_used=res.model,
        embedding_model=settings.EMBEDDING_MODEL,
        documents_used="[]",
    )
    db.add(db_session)
    db.commit()

    return res


@router.post("/ask/rag", response_model=AnswerResult)
async def ask_rag_only(payload: QARequest, db: Session = Depends(get_db)):
    """Answer a question strictly grounded on retrieved document context."""
    if not payload.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    model_config = {
        "backend": payload.backend or settings.DEFAULT_LLM_BACKEND,
        "model": payload.model,
    }

    res = await asyncio.to_thread(
        rag_pipeline.answer_with_rag,
        question=payload.question,
        top_k=payload.top_k,
        retrieval_mode=payload.retrieval_mode,
        use_reranking=payload.use_reranking,
        model_config=model_config,
        document_id=payload.document_id,
    )

    docs_used = list({c.filename for c in res.retrieved_chunks or [] if c.filename})

    db_session = QASession(
        id=str(uuid.uuid4()),
        question=payload.question,
        standard_answer=None,
        rag_answer=res.answer,
        standard_latency_ms=0.0,
        rag_latency_ms=res.latency_ms,
        top_k=payload.top_k,
        model_used=res.model,
        embedding_model=settings.EMBEDDING_MODEL,
        documents_used=json.dumps(docs_used),
    )
    db.add(db_session)
    db.commit()

    return res


@router.get("/sessions", response_model=QASessionListResponse)
async def list_qa_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List historical QA sessions with pagination."""
    query = db.query(QASession).order_by(QASession.session_time.desc())
    total = query.count()
    sessions = query.offset(skip).limit(limit).all()

    formatted: list[QASessionInfo] = []
    for s in sessions:
        docs = []
        try:
            docs = json.loads(s.documents_used or "[]")
        except Exception:
            docs = []

        formatted.append(
            QASessionInfo(
                id=s.id,
                question=s.question,
                standard_answer=s.standard_answer,
                rag_answer=s.rag_answer,
                standard_latency_ms=s.standard_latency_ms,
                rag_latency_ms=s.rag_latency_ms,
                top_k=s.top_k,
                model_used=s.model_used,
                embedding_model=s.embedding_model,
                session_time=s.session_time,
                documents_used=docs,
            )
        )

    return QASessionListResponse(total=total, sessions=formatted)


@router.get("/sessions/{session_id}")
async def get_qa_session(session_id: str, db: Session = Depends(get_db)):
    """Retrieve full details of a specific QA session, including associated evaluation metrics."""
    s = db.query(QASession).filter(QASession.id == session_id).first()
    if not s:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"QA session {session_id} not found")

    docs = []
    try:
        docs = json.loads(s.documents_used or "[]")
    except Exception:
        docs = []

    evals = [
        {
            "id": e.id,
            "metric_name": e.metric_name,
            "metric_value": e.metric_value,
            "is_reference_based": e.is_reference_based,
            "computed_at": e.computed_at,
            "notes": e.notes,
        }
        for e in s.evaluations
    ]

    return {
        "id": s.id,
        "question": s.question,
        "standard_answer": s.standard_answer,
        "rag_answer": s.rag_answer,
        "standard_latency_ms": s.standard_latency_ms,
        "rag_latency_ms": s.rag_latency_ms,
        "top_k": s.top_k,
        "model_used": s.model_used,
        "embedding_model": s.embedding_model,
        "session_time": s.session_time,
        "documents_used": docs,
        "evaluations": evals,
    }
