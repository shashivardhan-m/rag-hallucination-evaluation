"""API endpoints for system-wide performance analytics, hallucination trends, and model comparisons."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import (
    Document,
    DocumentChunk,
    EvaluationResult,
    Experiment,
    QASession,
)
from app.models.schemas import (
    AnalyticsSummary,
    MetricTimeSeriesResponse,
    ModelComparison,
    TimeSeriesPoint,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Aggregate high-level system metrics, counts, and average evaluation scores."""
    total_docs = db.query(Document).count()
    total_chunks = db.query(DocumentChunk).count()
    total_qa = db.query(QASession).count()
    total_evals = db.query(EvaluationResult).count()
    total_exps = db.query(Experiment).count()

    # Latencies
    std_lat = (
        db.query(func.avg(QASession.standard_latency_ms))
        .filter(QASession.standard_latency_ms > 0)
        .scalar()
        or 0.0
    )
    rag_lat = (
        db.query(func.avg(QASession.rag_latency_ms))
        .filter(QASession.rag_latency_ms > 0)
        .scalar()
        or 0.0
    )

    def _get_metric_avg(name: str) -> float:
        val = (
            db.query(func.avg(EvaluationResult.metric_value))
            .filter(EvaluationResult.metric_name == name)
            .scalar()
        )
        return round(float(val), 4) if val is not None else 0.0

    return AnalyticsSummary(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_qa_sessions=total_qa,
        total_evaluations=total_evals,
        total_experiments=total_exps,
        avg_standard_latency_ms=round(float(std_lat), 2),
        avg_rag_latency_ms=round(float(rag_lat), 2),
        avg_faithfulness=_get_metric_avg("faithfulness"),
        avg_hallucination_rate=_get_metric_avg("hallucination_rate"),
        avg_answer_relevance=_get_metric_avg("answer_relevance"),
        avg_answer_correctness=_get_metric_avg("answer_correctness"),
    )


@router.get("/model-comparison", response_model=ModelComparison)
async def compare_models(db: Session = Depends(get_db)):
    """Compare performance and hallucination metrics grouped by model used."""
    # Find distinct models used in sessions
    models = [r[0] for r in db.query(QASession.model_used).distinct().all() if r[0]]

    comparison: List[Dict[str, Any]] = []

    for m in models:
        session_query = db.query(QASession).filter(QASession.model_used == m)
        count = session_query.count()

        avg_std_lat = (
            db.query(func.avg(QASession.standard_latency_ms))
            .filter(QASession.model_used == m, QASession.standard_latency_ms > 0)
            .scalar()
            or 0.0
        )
        avg_rag_lat = (
            db.query(func.avg(QASession.rag_latency_ms))
            .filter(QASession.model_used == m, QASession.rag_latency_ms > 0)
            .scalar()
            or 0.0
        )

        # Evaluations for this model's sessions
        session_ids = [s.id for s in session_query.all()]
        metric_averages: Dict[str, float] = {}

        if session_ids:
            eval_rows = (
                db.query(
                    EvaluationResult.metric_name,
                    func.avg(EvaluationResult.metric_value),
                )
                .filter(EvaluationResult.qa_session_id.in_(session_ids))
                .group_by(EvaluationResult.metric_name)
                .all()
            )
            for m_name, avg_v in eval_rows:
                metric_averages[m_name] = round(float(avg_v), 4)

        comparison.append({
            "model_name": m,
            "session_count": count,
            "avg_standard_latency_ms": round(float(avg_std_lat), 2),
            "avg_rag_latency_ms": round(float(avg_rag_lat), 2),
            "faithfulness": metric_averages.get("faithfulness", 0.0),
            "hallucination_rate": metric_averages.get("hallucination_rate", 0.0),
            "answer_relevance": metric_averages.get("answer_relevance", 0.0),
            "answer_correctness": metric_averages.get("answer_correctness", 0.0),
        })

    return ModelComparison(models=comparison)


@router.get("/metrics-over-time", response_model=MetricTimeSeriesResponse)
async def get_metrics_over_time(
    metric_name: str = Query("faithfulness"),
    db: Session = Depends(get_db),
):
    """Retrieve chronological time series data points for an evaluation metric."""
    results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.metric_name == metric_name)
        .order_by(EvaluationResult.computed_at.asc())
        .limit(200)
        .all()
    )

    points = [
        TimeSeriesPoint(
            timestamp=r.computed_at.isoformat(),
            value=round(float(r.metric_value), 4),
        )
        for r in results
    ]

    return MetricTimeSeriesResponse(metric_name=metric_name, points=points)


@router.get("/hallucination-trend", response_model=MetricTimeSeriesResponse)
async def get_hallucination_trend(db: Session = Depends(get_db)):
    """Retrieve chronologically ordered hallucination rate measurements."""
    results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.metric_name == "hallucination_rate")
        .order_by(EvaluationResult.computed_at.asc())
        .limit(200)
        .all()
    )

    points = [
        TimeSeriesPoint(
            timestamp=r.computed_at.isoformat(),
            value=round(float(r.metric_value), 4),
        )
        for r in results
    ]

    return MetricTimeSeriesResponse(metric_name="hallucination_rate", points=points)
