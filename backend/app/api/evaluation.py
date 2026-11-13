"""API endpoints for evaluating answer faithfulness, hallucination rate, and retrieval metrics."""

import asyncio
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.evaluator import get_evaluator
from app.database import get_db
from app.models.db_models import EvaluationResult, QASession
from app.models.schemas import (
    EvaluationListResponse,
    EvaluationRequest,
    EvaluationResponse,
    MetricResult,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
evaluator = get_evaluator()

METRIC_DEFINITIONS = {
    "faithfulness": {
        "display_name": "Faithfulness",
        "description": "Measures whether the claims in the generated answer are strictly entailed by the retrieved context using NLI (DeBERTa-v3).",
        "is_reference_based": False,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "hallucination_rate": {
        "display_name": "Hallucination Rate",
        "description": "The proportion of claims made in the answer that cannot be grounded in the source context (1.0 - faithfulness).",
        "is_reference_based": False,
        "range": "[0.0, 1.0]",
        "higher_is_better": False,
    },
    "answer_relevance": {
        "display_name": "Answer Relevance",
        "description": "Cosine similarity between the dense embedding of the generated answer and the original user question.",
        "is_reference_based": False,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "context_precision": {
        "display_name": "Context Precision",
        "description": "Average semantic similarity between the user question and the retrieved context chunks.",
        "is_reference_based": False,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "retrieval_precision": {
        "display_name": "Retrieval Precision",
        "description": "The proportion of retrieved context chunks that exceed the relevance similarity threshold (0.40).",
        "is_reference_based": False,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "context_recall": {
        "display_name": "Context Recall",
        "description": "Cosine similarity between the concatenated retrieved context and the reference ground-truth answer.",
        "is_reference_based": True,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "retrieval_recall": {
        "display_name": "Retrieval Recall@K",
        "description": "The fraction of ground-truth relevant document chunk IDs that were successfully retrieved in top-K.",
        "is_reference_based": True,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "answer_correctness": {
        "display_name": "Answer Correctness",
        "description": "Composite score combining lexical overlap (ROUGE-L 40%) and semantic similarity (BERTScore 60%) against the reference answer.",
        "is_reference_based": True,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "rouge_l_f1": {
        "display_name": "ROUGE-L F1",
        "description": "Longest common subsequence F1 measure comparing generated answer to reference answer.",
        "is_reference_based": True,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
    "bertscore_f1": {
        "display_name": "BERTScore F1",
        "description": "Token-level contextual embedding similarity F1 comparing generated answer to reference answer.",
        "is_reference_based": True,
        "range": "[0.0, 1.0]",
        "higher_is_better": True,
    },
}


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(payload: EvaluationRequest, db: Session = Depends(get_db)):
    """
    Run comprehensive evaluation suite on an answer against provided context chunks
    and optional ground-truth reference answer. Persists results to DB if qa_session_id is provided.
    """
    if not payload.question.strip() or not payload.answer.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question and answer strings must not be empty.",
        )

    # If qa_session_id is provided, verify session exists
    if payload.qa_session_id:
        session = db.query(QASession).filter(QASession.id == payload.qa_session_id).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"QA session {payload.qa_session_id} not found.",
            )

    try:
        eval_output = await asyncio.to_thread(
            evaluator.evaluate_qa_pair,
            question=payload.question,
            answer=payload.answer,
            context_chunks=payload.context_chunks,
            reference=payload.reference_answer,
            retrieved_ids=payload.retrieved_ids,
            relevant_ids=payload.relevant_ids,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation execution failed: {str(e)}",
        )

    metrics = eval_output.get("metrics", {})
    details = eval_output.get("details", {})
    eval_batch_id = str(uuid.uuid4())

    # Persist metrics to database
    for m_name, m_val in metrics.items():
        definition = METRIC_DEFINITIONS.get(m_name, {})
        is_ref = definition.get("is_reference_based", False)

        db_eval = EvaluationResult(
            id=str(uuid.uuid4()),
            qa_session_id=payload.qa_session_id,
            metric_name=m_name,
            metric_value=float(m_val),
            is_reference_based=is_ref,
            notes=f"Evaluation batch: {eval_batch_id}",
        )
        db.add(db_eval)

    db.commit()

    return EvaluationResponse(
        evaluation_id=eval_batch_id,
        qa_session_id=payload.qa_session_id,
        metrics=metrics,
        details=details,
    )


@router.get("/metrics/definitions")
async def get_metrics_definitions():
    """Return dictionary of all evaluation metric definitions, formulas, and characteristics."""
    return METRIC_DEFINITIONS


@router.get("/results", response_model=EvaluationListResponse)
async def list_evaluation_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    metric_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """List stored evaluation metrics with optional filtering by metric name."""
    query = db.query(EvaluationResult).order_by(EvaluationResult.computed_at.desc())
    if metric_name:
        query = query.filter(EvaluationResult.metric_name == metric_name)

    total = query.count()
    results = query.offset(skip).limit(limit).all()

    return EvaluationListResponse(
        total=total,
        results=[MetricResult.model_validate(r) for r in results],
    )


@router.get("/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str, db: Session = Depends(get_db)):
    """Retrieve all evaluation metrics computed for a specific QA session."""
    session = db.query(QASession).filter(QASession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"QA session {session_id} not found",
        )

    evals = db.query(EvaluationResult).filter(EvaluationResult.qa_session_id == session_id).all()
    metrics_map = {e.metric_name: e.metric_value for e in evals}

    return {
        "session_id": session_id,
        "question": session.question,
        "metrics": metrics_map,
        "total_metrics_computed": len(evals),
    }
