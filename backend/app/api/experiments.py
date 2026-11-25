"""API endpoints for automated benchmarking experiments varying retrieval parameters, models, and ablations."""

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
from app.core.benchmarks import get_benchmark_registry
from app.core.evaluator import get_evaluator
from app.core.rag_pipeline import get_rag_pipeline
from app.core.vector_store import get_vector_store
from app.database import get_db
from app.models.db_models import Experiment
from app.models.schemas import (
    ExperimentListResponse,
    ExperimentResult,
    ExperimentRunRequest,
)
from app.utils.metrics_utils import (
    aggregate_metrics,
    aggregate_metrics_with_stats,
    compute_statistical_comparison,
)

router = APIRouter(prefix="/experiments", tags=["Experiments"])
settings = get_settings()
rag_pipeline = get_rag_pipeline()
evaluator = get_evaluator()
benchmark_registry = get_benchmark_registry()
vector_store = get_vector_store()


@router.get("/benchmarks/list")
async def list_available_benchmarks():
    """List standard benchmarks (RAGTruth, HaluEval) available for evaluation."""
    return benchmark_registry.list_benchmarks()


async def _run_experiment_pipeline(
    exp_id: str,
    name: str,
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the benchmark loop across parameter configurations, models, and statistical runs."""
    variable = config_dict.get("variable", "top_k")
    values = config_dict.get("values", [1, 3, 5])
    benchmark_id = config_dict.get("benchmark_dataset")
    num_runs = int(config_dict.get("num_runs", 1))

    questions = list(config_dict.get("questions") or [])
    ref_answers = list(config_dict.get("reference_answers") or [])
    benchmark_items = []

    # If benchmark dataset specified, load from standard registry
    if benchmark_id:
        try:
            benchmark_items = benchmark_registry.load_benchmark(benchmark_id)
            if not questions:
                questions = [it.question for it in benchmark_items]
                ref_answers = [it.ground_truth_answer for it in benchmark_items]

            # Ingest benchmark passages if vector store has no documents
            if len(vector_store.get_all_chunks()) == 0:
                bench_docs = benchmark_registry.export_as_documents(benchmark_id)
                # Compute embeddings & add
                from app.core.embeddings import get_embedding_manager
                em = get_embedding_manager()
                texts = [d["content"] for d in bench_docs]
                embs = em.get_embeddings(texts)
                vector_store.add_chunks(bench_docs, embs, document_id=f"benchmark_{benchmark_id}")
        except Exception as e:
            pass

    if not ref_answers or len(ref_answers) < len(questions):
        ref_answers = ref_answers + [None] * (len(questions) - len(ref_answers))

    default_top_k = int(config_dict.get("default_top_k", 5))
    default_mode = str(config_dict.get("default_retrieval_mode", "semantic"))
    default_rerank = bool(config_dict.get("default_use_reranking", False))
    backend = config_dict.get("backend") or settings.DEFAULT_LLM_BACKEND
    model = config_dict.get("model") or settings.DEFAULT_GROQ_MODEL

    condition_results: List[Dict[str, Any]] = []
    baseline_faithfulness_series: List[float] = []

    for c_idx, val in enumerate(values):
        curr_top_k = default_top_k
        curr_mode = default_mode
        curr_rerank = default_rerank
        curr_model = model
        curr_backend = backend
        context_condition = "retrieved"

        if variable == "top_k":
            curr_top_k = int(val)
        elif variable == "retrieval_mode":
            curr_mode = str(val)
        elif variable == "reranking":
            curr_rerank = bool(val)
        elif variable == "llm_model":
            curr_model = str(val)
        elif variable == "context_relevance":
            context_condition = str(val).lower()
        elif variable == "chunk_size":
            curr_top_k = default_top_k

        model_config = {"backend": curr_backend, "model": curr_model}

        query_records: List[Dict[str, Any]] = []
        metrics_list: List[Dict[str, float]] = []
        latencies: List[float] = []
        taxonomy_accumulator: Dict[str, List[float]] = {}

        # Repeated runs for statistical confidence intervals
        for run_idx in range(num_runs):
            for q_idx, question in enumerate(questions):
                ref = ref_answers[q_idx] if q_idx < len(ref_answers) else None
                benchmark_item = benchmark_items[q_idx] if q_idx < len(benchmark_items) else None

                # Determine context: controlled perturbation or live retrieval
                if context_condition in ("gold", "distractor", "irrelevant", "none") and benchmark_item:
                    perturbed_chunks = benchmark_registry.get_perturbed_contexts(benchmark_item, context_condition)
                    # Directly query LLM with perturbed context
                    if not perturbed_chunks:
                        rag_res = await asyncio.to_thread(
                            rag_pipeline.answer_standard,
                            question=question,
                            model_config=model_config,
                        )
                        context_texts = []
                    else:
                        from app.utils.text_utils import format_context_for_prompt
                        from app.core.rag_pipeline import RAG_SYSTEM_PROMPT
                        ctx_str = "\n\n".join(perturbed_chunks)
                        rag_prompt = (
                            f"Here is the verified context from our document store:\n"
                            f"----------------------------------------\n"
                            f"{ctx_str}\n"
                            f"----------------------------------------\n\n"
                            f"User Question: {question}\n\n"
                            f"Provide your answer grounded strictly in the context above:"
                        )
                        client = rag_pipeline._resolve_client(model_config)
                        ans, lat = await asyncio.to_thread(
                            client.generate,
                            prompt=rag_prompt,
                            system_prompt=RAG_SYSTEM_PROMPT,
                            temperature=0.0,
                        )
                        context_texts = perturbed_chunks
                        from app.models.schemas import AnswerResult
                        rag_res = AnswerResult(
                            answer=ans,
                            latency_ms=lat,
                            model=client.model,
                            backend=client.backend,
                            retrieved_chunks=None,
                        )
                else:
                    # Standard live RAG retrieval pipeline
                    rag_res = await asyncio.to_thread(
                        rag_pipeline.answer_with_rag,
                        question=question,
                        top_k=curr_top_k,
                        retrieval_mode=curr_mode,
                        use_reranking=curr_rerank,
                        model_config=model_config,
                    )
                    context_texts = [c.content for c in rag_res.retrieved_chunks or []]

                # Evaluate metrics & fine-grained taxonomy
                eval_output = await asyncio.to_thread(
                    evaluator.evaluate_qa_pair,
                    question=question,
                    answer=rag_res.answer,
                    context_chunks=context_texts,
                    reference=ref,
                )

                m = eval_output.get("metrics", {})
                tax = eval_output.get("error_taxonomy", {})
                metrics_list.append(m)
                latencies.append(rag_res.latency_ms)

                for t_name, t_val in tax.items():
                    taxonomy_accumulator.setdefault(t_name, []).append(float(t_val))

                if run_idx == 0:  # store primary records from first pass
                    query_records.append({
                        "question": question,
                        "reference": ref,
                        "answer": rag_res.answer,
                        "latency_ms": rag_res.latency_ms,
                        "retrieved_count": len(context_texts),
                        "metrics": m,
                        "error_taxonomy": tax,
                        "claims_analysis": eval_output.get("details", {}).get("claims_analysis", []),
                    })

        avg_m = aggregate_metrics(metrics_list)
        stats_m = aggregate_metrics_with_stats(metrics_list, confidence=0.95)
        avg_lat = round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0

        # Mean taxonomy distribution
        avg_taxonomy = {}
        for t_k, t_list in taxonomy_accumulator.items():
            avg_taxonomy[t_k] = round(float(sum(t_list) / max(1, len(t_list))), 4)

        # Statistical comparison against baseline (condition 0)
        curr_faithfulness = [m.get("faithfulness", 0.0) for m in metrics_list]
        if c_idx == 0:
            baseline_faithfulness_series = curr_faithfulness
            stat_comparison = {
                "is_baseline": True,
                "p_value": 1.0,
                "t_statistic": 0.0,
                "is_significant_p05": False,
                "is_significant_p01": False,
                "cohens_d": 0.0,
            }
        else:
            stat_comparison = compute_statistical_comparison(
                baseline_values=baseline_faithfulness_series,
                treatment_values=curr_faithfulness,
            )
            stat_comparison["is_baseline"] = False

        condition_results.append({
            "variable": variable,
            "condition_value": val,
            "runs_executed": num_runs,
            "average_latency_ms": avg_lat,
            "average_metrics": avg_m,
            "statistical_metrics": stats_m,
            "significance_vs_baseline": stat_comparison,
            "error_taxonomy": avg_taxonomy,
            "queries": query_records,
        })

    return {
        "experiment_id": exp_id,
        "name": name,
        "variable_tested": variable,
        "benchmark_dataset": benchmark_id,
        "num_runs": num_runs,
        "total_queries_tested": len(questions) * len(values) * num_runs,
        "conditions": condition_results,
    }


@router.post("/run", response_model=ExperimentResult, status_code=status.HTTP_201_CREATED)
async def run_experiment(payload: ExperimentRunRequest, db: Session = Depends(get_db)):
    """
    Launch an empirical benchmark varying a hyperparameter (retrieval_mode, reranking, top_k, llm_model, context_relevance)
    across standard benchmark questions (RAGTruth, HaluEval) or custom queries.
    """
    config_dict = payload.config.model_dump()
    if not config_dict.get("questions") and not config_dict.get("benchmark_dataset"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment must specify at least one question or a standard benchmark_dataset ('ragtruth', 'halueval').",
        )
    if not config_dict.get("values"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment must contain at least one variable value to test.",
        )

    exp_id = str(uuid.uuid4())
    q_count = len(config_dict.get("questions") or []) or 5
    num_queries = q_count * len(config_dict.get("values")) * int(config_dict.get("num_runs", 1))

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

        # Save to disk
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

    disk_file = Path(settings.EXPERIMENTS_PATH) / f"{exp_id}.json"
    if disk_file.exists():
        disk_file.unlink()

    db.delete(e)
    db.commit()

    return {"message": f"Experiment {exp_id} deleted successfully"}
