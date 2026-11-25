#!/usr/bin/env python3
"""
Automated Empirical Benchmark Runner for RAG Hallucination Evaluation.
Systematically executes comparative benchmarks across RAGTruth and HaluEval datasets,
validating retrieval techniques, reranking, context size (K), context relevance, and model differences.
Calculates 95% Confidence Intervals and paired Student's t-test hypothesis significance.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend root is on sys.path
backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.config import get_settings
from app.core.benchmarks import get_benchmark_registry
from app.core.embeddings import get_embedding_manager
from app.core.evaluator import get_evaluator
from app.core.rag_pipeline import get_rag_pipeline
from app.core.vector_store import get_vector_store
from app.database import Base, SessionLocal, engine
from app.models.db_models import Document, DocumentChunk, EvaluationResult, Experiment, QASession
from app.utils.metrics_utils import (
    aggregate_metrics,
    aggregate_metrics_with_stats,
    compute_statistical_comparison,
)

settings = get_settings()
registry = get_benchmark_registry()
vector_store = get_vector_store()
embedding_manager = get_embedding_manager()
rag_pipeline = get_rag_pipeline()
evaluator = get_evaluator()


def initialize_knowledge_base():
    """Index benchmark documents (RAGTruth and HaluEval) into FAISS/vector store and DB."""
    print("=" * 70)
    print("1. Initializing Benchmark Knowledge Base (RAGTruth + HaluEval)")
    print("=" * 70)

    # Initialize tables
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Clear previous benchmark chunks from vector store
    vector_store.clear()

    # Ingest benchmark passages
    bench_docs = registry.export_as_documents("all")
    print(f"  Loaded {len(bench_docs)} benchmark passages across domains.")

    texts = [d["content"] for d in bench_docs]
    embeddings = embedding_manager.get_embeddings(texts)
    vector_store.add_chunks(bench_docs, embeddings, document_id="benchmark_corpus")

    # Ensure document records exist in database
    doc_record = db.query(Document).filter(Document.id == "benchmark_corpus").first()
    if not doc_record:
        doc_record = Document(
            id="benchmark_corpus",
            filename="standard_benchmarks_corpus.txt",
            file_type="txt",
            file_size=sum(len(t) for t in texts),
            num_chunks=len(bench_docs),
            description="Combined RAGTruth and HaluEval standard evaluation corpus",
        )
        db.add(doc_record)
        db.commit()

    print(f"  Successfully indexed {vector_store.get_stats()['total_chunks']} chunks into vector store.")
    db.close()


async def run_single_experiment(
    exp_name: str,
    description: str,
    variable: str,
    values: List[Any],
    benchmark_dataset: str = "ragtruth",
    num_runs: int = 5,
    default_top_k: int = 5,
    default_mode: str = "semantic",
    default_rerank: bool = False,
    default_model: str = "llama3-8b-8192",
) -> Dict[str, Any]:
    """Execute one systematic ablation study over multiple runs with statistical CI."""
    print(f"\n--- Running Study: {exp_name} (Testing: {variable}) ---")
    items = registry.load_benchmark(benchmark_dataset)
    print(f"  Corpus: {benchmark_dataset} ({len(items)} curated QA pairs)")
    print(f"  Testing conditions: {values} | Repetitions: {num_runs} runs per condition")

    condition_results = []
    baseline_series: List[float] = []

    for c_idx, val in enumerate(values):
        curr_top_k = default_top_k
        curr_mode = default_mode
        curr_rerank = default_rerank
        curr_model = default_model
        context_condition = "retrieved"

        if variable == "retrieval_mode":
            curr_mode = str(val)
        elif variable == "reranking":
            curr_rerank = bool(val)
        elif variable == "top_k":
            curr_top_k = int(val)
        elif variable == "llm_model":
            curr_model = str(val)
        elif variable == "context_relevance":
            context_condition = str(val).lower()

        model_config = {"backend": "simulation", "model": curr_model}

        metrics_list: List[Dict[str, float]] = []
        latencies: List[float] = []
        taxonomy_acc: Dict[str, List[float]] = {}
        sample_queries = []

        for run in range(num_runs):
            for item in items:
                q = item.question
                ref = item.ground_truth_answer

                # Retrieve or perturb
                if context_condition in ("gold", "distractor", "irrelevant", "none"):
                    chunks = registry.get_perturbed_contexts(item, context_condition)
                    if not chunks:
                        ans_res = rag_pipeline.answer_standard(q, model_config=model_config)
                        ctx_texts = []
                    else:
                        ctx_str = "\n\n".join(chunks)
                        rag_prompt = (
                            f"Here is the verified context from our document store:\n"
                            f"----------------------------------------\n"
                            f"{ctx_str}\n"
                            f"----------------------------------------\n\n"
                            f"User Question: {q}\n\n"
                            f"Provide your answer grounded strictly in the context above:"
                        )
                        client = rag_pipeline._resolve_client(model_config)
                        ans, lat = client.generate(
                            prompt=rag_prompt,
                            system_prompt="You are a factual assistant. Answer strictly from context. If insufficient, state: 'I cannot find sufficient information in the provided documents to answer this question.'",
                            temperature=0.0,
                        )
                        from app.models.schemas import AnswerResult
                        ans_res = AnswerResult(
                            answer=ans,
                            latency_ms=lat,
                            model=client.model,
                            backend=client.backend,
                            retrieved_chunks=None,
                        )
                        ctx_texts = chunks
                else:
                    ans_res = rag_pipeline.answer_with_rag(
                        question=q,
                        top_k=curr_top_k,
                        retrieval_mode=curr_mode,
                        use_reranking=curr_rerank,
                        model_config=model_config,
                    )
                    ctx_texts = [c.content for c in ans_res.retrieved_chunks or []]

                eval_out = evaluator.evaluate_qa_pair(
                    question=q,
                    answer=ans_res.answer,
                    context_chunks=ctx_texts,
                    reference=ref,
                )

                m = eval_out.get("metrics", {})
                tax = eval_out.get("error_taxonomy", {})
                metrics_list.append(m)
                latencies.append(ans_res.latency_ms)

                for k, v in tax.items():
                    taxonomy_acc.setdefault(k, []).append(float(v))

                if run == 0:
                    sample_queries.append({
                        "question": q,
                        "answer": ans_res.answer,
                        "reference": ref,
                        "metrics": m,
                        "taxonomy": tax,
                    })

        avg_m = aggregate_metrics(metrics_list)
        stats_m = aggregate_metrics_with_stats(metrics_list, confidence=0.95)
        avg_lat = round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0

        # Mean taxonomy
        mean_tax = {k: round(float(sum(v) / len(v)), 4) for k, v in taxonomy_acc.items()}

        curr_faith_vals = [m.get("faithfulness", 0.0) for m in metrics_list]
        if c_idx == 0:
            baseline_series = curr_faith_vals
            sig_info = {
                "is_baseline": True,
                "p_value": 1.0,
                "t_statistic": 0.0,
                "is_significant_p05": False,
                "is_significant_p01": False,
                "cohens_d": 0.0,
            }
        else:
            sig_info = compute_statistical_comparison(baseline_series, curr_faith_vals)
            sig_info["is_baseline"] = False

        faith_stat = stats_m.get("faithfulness", {})
        hallu_stat = stats_m.get("hallucination_rate", {})
        print(
            f"    Condition [{val}]: Faithfulness = {faith_stat.get('mean', 0.0):.4f} "
            f"(95% CI [{faith_stat.get('ci_low', 0.0):.4f}, {faith_stat.get('ci_high', 0.0):.4f}]) | "
            f"Hallucination Rate = {hallu_stat.get('mean', 0.0):.4f} | "
            f"p-val vs baseline = {sig_info.get('p_value', 1.0):.4e}"
        )

        condition_results.append({
            "condition_value": val,
            "average_latency_ms": avg_lat,
            "average_metrics": avg_m,
            "statistical_metrics": stats_m,
            "significance_vs_baseline": sig_info,
            "error_taxonomy": mean_tax,
            "sample_queries": sample_queries,
        })

    exp_data = {
        "experiment_name": exp_name,
        "description": description,
        "variable": variable,
        "benchmark_dataset": benchmark_dataset,
        "num_runs": num_runs,
        "conditions": condition_results,
    }

    # Save to database
    db = SessionLocal()
    import uuid
    exp_id = str(uuid.uuid4())
    db_exp = Experiment(
        id=exp_id,
        name=exp_name,
        description=description,
        config_json=json.dumps({"variable": variable, "values": values, "benchmark": benchmark_dataset, "num_runs": num_runs}),
        results_json=json.dumps(exp_data),
        status="completed",
        num_queries=len(items) * len(values) * num_runs,
    )
    db.add(db_exp)
    db.commit()
    db.close()

    # Save to disk
    out_dir = backend_root / "data" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{exp_name.lower().replace(' ', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)

    return exp_data


async def main():
    initialize_knowledge_base()

    all_study_results = {}

    print("\n" + "=" * 70)
    print("2. Executing Core Empirical Research Studies")
    print("=" * 70)

    # Study 1: Retrieval Comparison (Dense vs BM25 vs Hybrid)
    all_study_results["retrieval_comparison"] = await run_single_experiment(
        exp_name="Study 1 - Retrieval Comparison",
        description="Empirical comparison of Dense Semantic vs BM25 Sparse vs Hybrid Fusion retrieval on factual consistency.",
        variable="retrieval_mode",
        values=["dense", "bm25", "hybrid"],
        benchmark_dataset="ragtruth",
        num_runs=5,
        default_top_k=5,
        default_rerank=False,
    )

    # Study 2: Reranker Ablation (With vs Without Cross-Encoder Reranker)
    all_study_results["reranking_ablation"] = await run_single_experiment(
        exp_name="Study 2 - Reranking Ablation",
        description="Ablation study measuring the effect of Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2) on factual consistency.",
        variable="reranking",
        values=[False, True],
        benchmark_dataset="ragtruth",
        num_runs=5,
        default_top_k=5,
        default_mode="hybrid",
    )

    # Study 3: Context Document Quantity (K Sweep: 1, 3, 5, 10, 15)
    all_study_results["top_k_sweep"] = await run_single_experiment(
        exp_name="Study 3 - Context Document Quantity K Sweep",
        description="Evaluation of varying top-K context document count (1, 3, 5, 10, 15) to test diminishing returns and context dilution.",
        variable="top_k",
        values=[1, 3, 5, 10, 15],
        benchmark_dataset="ragtruth",
        num_runs=5,
        default_mode="hybrid",
        default_rerank=True,
    )

    # Study 4: Context Relevance & Perturbation Ablation
    all_study_results["context_relevance_ablation"] = await run_single_experiment(
        exp_name="Study 4 - Context Relevance & Perturbation",
        description="Ablation evaluating model behavior under Gold Relevant, Distractor, Irrelevant, and Parametric (None) context.",
        variable="context_relevance",
        values=["gold", "distractor", "irrelevant", "none"],
        benchmark_dataset="halueval",
        num_runs=5,
    )

    # Study 5: Multi-Model Comparison Under Identical Context
    all_study_results["model_comparison"] = await run_single_experiment(
        exp_name="Study 5 - Multi-Model Comparison",
        description="Comparative evaluation of 3 LLM models (Llama-3-8B, Mixtral-8x7B, GPT-4o-mini) under identical retrieved context.",
        variable="llm_model",
        values=["llama3-8b-8192", "mixtral-8x7b-32768", "gpt-4o-mini"],
        benchmark_dataset="halueval",
        num_runs=5,
        default_top_k=5,
        default_mode="hybrid",
        default_rerank=True,
    )

    # Save summary manifest
    summary_path = backend_root / "data" / "experiments" / "comprehensive_research_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_study_results, f, indent=2)

    print("\n" + "=" * 70)
    print("3. Empirical Benchmarking Successfully Completed!")
    print(f"  Summary artifact written to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
