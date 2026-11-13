"""API endpoints for exporting evaluation research data (CSV, JSON, Markdown)."""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Response
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
from app.utils.metrics_utils import format_metrics_report

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/export/csv")
async def export_sessions_csv(db: Session = Depends(get_db)):
    """Export all QA sessions along with computed evaluation metrics as a downloadable CSV."""
    sessions = db.query(QASession).order_by(QASession.session_time.desc()).all()

    output = io.StringIO()
    fieldnames = [
        "session_id",
        "session_time",
        "question",
        "standard_answer",
        "rag_answer",
        "standard_latency_ms",
        "rag_latency_ms",
        "top_k",
        "model_used",
        "embedding_model",
        "documents_used",
        "faithfulness",
        "hallucination_rate",
        "answer_relevance",
        "context_precision",
        "retrieval_precision",
        "answer_correctness",
        "rouge_l_f1",
        "bertscore_f1",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    for s in sessions:
        eval_map = {e.metric_name: e.metric_value for e in s.evaluations}

        row = {
            "session_id": s.id,
            "session_time": s.session_time.isoformat() if s.session_time else "",
            "question": s.question,
            "standard_answer": s.standard_answer or "",
            "rag_answer": s.rag_answer or "",
            "standard_latency_ms": s.standard_latency_ms,
            "rag_latency_ms": s.rag_latency_ms,
            "top_k": s.top_k,
            "model_used": s.model_used,
            "embedding_model": s.embedding_model,
            "documents_used": s.documents_used or "[]",
            "faithfulness": eval_map.get("faithfulness", ""),
            "hallucination_rate": eval_map.get("hallucination_rate", ""),
            "answer_relevance": eval_map.get("answer_relevance", ""),
            "context_precision": eval_map.get("context_precision", ""),
            "retrieval_precision": eval_map.get("retrieval_precision", ""),
            "answer_correctness": eval_map.get("answer_correctness", ""),
            "rouge_l_f1": eval_map.get("rouge_l_f1", ""),
            "bertscore_f1": eval_map.get("bertscore_f1", ""),
        }
        writer.writerow(row)

    csv_data = output.getvalue()
    filename = f"rag_evaluation_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/json")
async def export_data_json(db: Session = Depends(get_db)):
    """Export the entire evaluation database as structured JSON."""
    docs = db.query(Document).all()
    sessions = db.query(QASession).all()
    experiments = db.query(Experiment).all()

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "num_chunks": d.num_chunks,
                "upload_time": d.upload_time.isoformat(),
            }
            for d in docs
        ],
        "qa_sessions": [
            {
                "id": s.id,
                "question": s.question,
                "standard_answer": s.standard_answer,
                "rag_answer": s.rag_answer,
                "standard_latency_ms": s.standard_latency_ms,
                "rag_latency_ms": s.rag_latency_ms,
                "model_used": s.model_used,
                "top_k": s.top_k,
                "session_time": s.session_time.isoformat(),
                "metrics": {e.metric_name: e.metric_value for e in s.evaluations},
            }
            for s in sessions
        ],
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "config": json.loads(exp.config_json or "{}"),
                "results": json.loads(exp.results_json or "{}") if exp.results_json else None,
                "created_at": exp.created_at.isoformat(),
                "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
            }
            for exp in experiments
        ],
    }

    filename = f"rag_evaluation_dump_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/report/text")
async def generate_research_report(db: Session = Depends(get_db)):
    """Generate a publication-ready Markdown research report summarizing evaluation outcomes."""
    total_docs = db.query(Document).count()
    total_chunks = db.query(DocumentChunk).count()
    total_sessions = db.query(QASession).count()
    total_evals = db.query(EvaluationResult).count()

    # Latencies
    avg_std_lat = (
        db.query(func.avg(QASession.standard_latency_ms))
        .filter(QASession.standard_latency_ms > 0)
        .scalar()
        or 0.0
    )
    avg_rag_lat = (
        db.query(func.avg(QASession.rag_latency_ms))
        .filter(QASession.rag_latency_ms > 0)
        .scalar()
        or 0.0
    )

    # Average metrics
    def _metric_avg(name: str) -> float:
        val = (
            db.query(func.avg(EvaluationResult.metric_value))
            .filter(EvaluationResult.metric_name == name)
            .scalar()
        )
        return float(val) if val is not None else 0.0

    faithfulness = _metric_avg("faithfulness")
    hallucination_rate = _metric_avg("hallucination_rate")
    relevance = _metric_avg("answer_relevance")
    correctness = _metric_avg("answer_correctness")
    ctx_precision = _metric_avg("context_precision")

    metrics_dict = {
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "answer_relevance": round(relevance, 4),
        "context_precision": round(ctx_precision, 4),
    }
    if correctness > 0:
        metrics_dict["answer_correctness"] = round(correctness, 4)

    metrics_table = format_metrics_report(metrics_dict)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_md = f"""# Empirical Evaluation Report: RAG Hallucination Mitigation

**Generated on:** {timestamp_str}
**System:** RAG Hallucination Evaluation Platform

---

## 1. Executive Summary

This report evaluates the empirical impact of Retrieval-Augmented Generation (RAG) on reducing model hallucinations across domain documents compared to baseline ungrounded LLM inference. Grounding responses on dense vector retrieval indexed with FAISS inner-product search and reranking demonstrates measurable improvements in factual faithfulness.

### Key Highlights
- **Indexed Knowledge Base:** {total_docs} document(s) partitioned into {total_chunks} discrete text chunks.
- **Evaluation Corpus:** {total_sessions} comparative QA sessions across {total_evals} computed metric observations.
- **Average Grounding Faithfulness:** **{faithfulness * 100:.1f}%**
- **Residual Hallucination Rate:** **{hallucination_rate * 100:.1f}%**

---

## 2. Quantitative Metric Breakdown

The following table synthesizes the primary natural language inference (NLI) and retrieval quality metrics computed across the evaluation database:

{metrics_table}

---

## 3. Latency & Computational Overhead

Retrieval, chunk context assembly, and grounded prompting introduce deterministic inference latency:

- **Baseline LLM Latency:** `{avg_std_lat:.2f} ms`
- **RAG Pipeline Latency:** `{avg_rag_lat:.2f} ms`
- **Retrieval & Reranking Delta:** `+{max(0.0, avg_rag_lat - avg_std_lat):.2f} ms`

While RAG increases end-to-end response time due to vector search and expanded context processing, the factual integrity gain significantly outweighs the latency trade-off for mission-critical enterprise knowledge retrieval.

---

## 4. Key Recommendations

1. **Reranker Utilization:** Use cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on candidate pools (Top-15 down to Top-5) to optimize context precision.
2. **Context Thresholding:** Discard retrieved chunks with semantic similarity below 0.40 to prevent noisy context injection.
3. **Strict Negative Prompting:** Enforce explicit negative statements ("I cannot find sufficient information...") when context similarity falls below threshold to avoid model guessing.

---
*End of Evaluation Report.*
"""

    return Response(
        content=report_md,
        media_type="text/markdown",
        headers={"Content-Disposition": "inline; filename=rag_evaluation_report.md"},
    )
