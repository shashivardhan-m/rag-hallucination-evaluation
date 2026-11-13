"""API endpoints for automated benchmarking experiments varying retrieval parameters."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.evaluator import get_evaluator
from app.core.rag_pipeline import get_rag_pipeline
from app.database import get_db
from app.models.db_models import Experiment
from app.models.schemas import (
    ExperimentListResponse,
    ExperimentResult,
    ExperimentRunRequest,
)
from app.utils.metrics_utils import aggregate_metrics

router = APIRouter(prefix="/experiments", tags=["Experiments"])
settings = get_settings()
rag_pipeline = get_rag_pipeline()
evaluator = get_evaluator()


async def _run_experiment_pipeline(
    exp_id: str,
    name: str,
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the benchmark loop across all parameter configurations and questions."""
    variable = config_dict.get("variable", "top_k")
    values = config_dict.get("values", [1, 3, 5])
    questions = config_dict.get("questions", [])
    ref_answers = config_dict.get("reference_answers") or [None] * len(questions)

    default_top_k = int(config_dict.get("default_top_k", 5))
    default_mode = str(config_dict.get("default_retrieval_mode", "semantic"))
    default_rerank = bool(config_dict.get("default_use_reranking", False))
    backend = config_dict.get("backend") or settings.DEFAULT_LLM_BACKEND
    model = config_dict.get("model") or settings.DEFAULT_GROQ_MODEL

    model_config = {"backend": backend, "model": model}

    condition_results: List[Dict[str, Any]] = []

    for val in values:
        curr_top_k = default_top_k
        curr_mode = default_mode
        curr_rerank = default_rerank

        if variable == "top_k":
            curr_top_k = int(val)
        elif variable == "retrieval_mode":
            curr_mode = str(val)
        elif variable == "reranking":
            curr_rerank = bool(val)

        query_records: List[Dict[str, Any]] = []
        metrics_list: List[Dict[str, float]] = []
        latencies: List[float] = []

        for q_idx, question in enumerate(questions):
            ref = ref_answers[q_idx] if q_idx < len(ref_answers) else None

            # Generate RAG response
            rag_res = await asyncio.to_thread(
                rag_pipeline.answer_with_rag,
                question=question,
                top_k=curr_top_k,
                retrieval_mode=curr_mode,
                use_reranking=curr_rerank,
                model_config=model_config,
            )

            context_texts = [c.content for c in rag_res.retrieved_chunks or []]

            # Evaluate metrics
            eval_output = await asyncio.to_thread(
                evaluator.evaluate_qa_pair,
                question=question,
                answer=rag_res.answer,
                context_chunks=context_texts,
                reference=ref,
            )

            m = eval_output.get("metrics", {})
            metrics_list.append(m)
            latencies.append(rag_res.latency_ms)

            query_records.append({
                "question": question,
                "reference": ref,
                "answer": rag_res.answer,
                "latency_ms": rag_res.latency_ms,
                "retrieved_count": len(rag_res.retrieved_chunks or []),
                "metrics": m,
            })

        avg_m = aggregate_metrics(metrics_list)
        avg_lat = round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0

        condition_results.append({
            "variable": variable,
            "condition_value": val,
            "average_latency_ms": avg_lat,
            "average_metrics": avg_m,
            "queries": query_records,
        })

    return {
        "experiment_id": exp_id,
        "name": name,
        "variable_tested": variable,
        "total_queries_tested": len(questions) * len(values),
        "conditions": condition_results,
    }


@router.post("/run", response_model=ExperimentResult, status_code=status.HTTP_201_CREATED)
async def run_experiment(payload: ExperimentRunRequest, db: Session = Depends(get_db)):
    """
    Launch an empirical benchmark varying a hyperparameter (top_k, retrieval_mode, reranking)
    across a set of test questions, recording evaluation metrics for each condition.
    """
    if not payload.config.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment must contain at least one question.",
        )
    if not payload.config.values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment must contain at least one variable value to test.",
        )

    exp_id = str(uuid.uuid4())
    config_dict = payload.config.model_dump()
    num_queries = len(payload.config.questions) * len(payload.config.values)

    db_exp = Experiment(
        id=exp_id,
        name=payload.name,
        description=payload.description,
        config_json=json.dumps(config_dict),
        status="running",
        num_queries=num_queries,
    )
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)

    try:
        results = await _run_experiment_pipeline(
            exp_id=exp_id,
            name=payload.name,
            config_dict=config_dict,
        )

        db_exp.status = "completed"
        db_exp.completed_at = datetime.now(timezone.utc)
        db_exp.results_json = json.dumps(results)
        db.commit()
        db.refresh(db_exp)

        # Save to disk as well
        exp_dir = Path(settings.EXPERIMENTS_PATH)
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / f"{exp_id}.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        db_exp.status = "failed"
        db_exp.completed_at = datetime.now(timezone.utc)
        db_exp.results_json = json.dumps({"error": str(e)})
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment benchmark execution failed: {str(e)}",
        )

    return ExperimentResult(
        id=db_exp.id,
        name=db_exp.name,
        description=db_exp.description,
        status=db_exp.status,
        config=config_dict,
        results=results,
        created_at=db_exp.created_at,
        completed_at=db_exp.completed_at,
        num_queries=db_exp.num_queries,
    )


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieve all historical experiment runs."""
    query = db.query(Experiment).order_by(Experiment.created_at.desc())
    total = query.count()
    exps = query.offset(skip).limit(limit).all()

    formatted: List[ExperimentResult] = []
    for e in exps:
        cfg = json.loads(e.config_json or "{}")
        res = json.loads(e.results_json or "{}") if e.results_json else None
        formatted.append(
            ExperimentResult(
                id=e.id,
                name=e.name,
                description=e.description,
                status=e.status,
                config=cfg,
                results=res,
                created_at=e.created_at,
                completed_at=e.completed_at,
                num_queries=e.num_queries,
            )
        )

    return ExperimentListResponse(total=total, experiments=formatted)


@router.get("/{exp_id}", response_model=ExperimentResult)
async def get_experiment(exp_id: str, db: Session = Depends(get_db)):
    """Retrieve detailed results and question-level breakdowns for an experiment."""
    e = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id} not found")

    cfg = json.loads(e.config_json or "{}")
    res = json.loads(e.results_json or "{}") if e.results_json else None

    return ExperimentResult(
        id=e.id,
        name=e.name,
        description=e.description,
        status=e.status,
        config=cfg,
        results=res,
        created_at=e.created_at,
        completed_at=e.completed_at,
        num_queries=e.num_queries,
    )


@router.delete("/{exp_id}", status_code=status.HTTP_200_OK)
async def delete_experiment(exp_id: str, db: Session = Depends(get_db)):
    """Delete an experiment and its disk artifacts."""
    e = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id} not found")

    # Delete disk file if present
    disk_file = Path(settings.EXPERIMENTS_PATH) / f"{exp_id}.json"
    if disk_file.exists():
        disk_file.unlink()

    db.delete(e)
    db.commit()

    return {"message": f"Experiment {exp_id} deleted successfully"}
