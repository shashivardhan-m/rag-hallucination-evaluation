"""Unit tests for RAGTruth & HaluEval benchmark loaders, statistical validation, and error taxonomy."""

import pytest
import numpy as np
from app.core.benchmarks import get_benchmark_registry, BenchmarkRegistry
from app.core.evaluator import HallucinationClassifier, get_evaluator
from app.utils.metrics_utils import (
    compute_confidence_interval,
    compute_bootstrap_ci,
    compute_statistical_comparison,
    aggregate_metrics_with_stats,
    compute_rouge_l_native,
)


def test_benchmark_loader():
    registry = get_benchmark_registry()
    benchmarks = registry.list_benchmarks()
    assert len(benchmarks) >= 2
    b_ids = [b["id"] for b in benchmarks]
    assert "ragtruth" in b_ids
    assert "halueval" in b_ids

    # Load RAGTruth items
    ragtruth_items = registry.load_benchmark("ragtruth")
    assert len(ragtruth_items) >= 5
    first = ragtruth_items[0]
    assert len(first.gold_context) > 0
    assert len(first.question) > 0
    assert len(first.ground_truth_answer) > 0

    # Perturbations
    gold_ctx = registry.get_perturbed_contexts(first, "gold")
    distractor_ctx = registry.get_perturbed_contexts(first, "distractor")
    irrelevant_ctx = registry.get_perturbed_contexts(first, "irrelevant")
    empty_ctx = registry.get_perturbed_contexts(first, "none")

    assert len(gold_ctx) == len(first.gold_context)
    assert len(distractor_ctx) > len(gold_ctx)
    assert len(irrelevant_ctx) > 0
    assert len(empty_ctx) == 0


def test_statistical_confidence_interval():
    # Constant values
    ci_const = compute_confidence_interval([0.8, 0.8, 0.8, 0.8, 0.8])
    assert ci_const["mean"] == 0.8
    assert ci_const["std"] == 0.0
    assert ci_const["ci_low"] == 0.8
    assert ci_const["ci_high"] == 0.8

    # Varied values
    values = [0.72, 0.78, 0.85, 0.81, 0.79]
    ci = compute_confidence_interval(values, confidence=0.95)
    assert ci["n"] == 5
    assert 0.70 < ci["mean"] < 0.85
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["std"] > 0


def test_paired_ttest_comparison():
    # Baseline vs improved treatment
    baseline = [0.45, 0.50, 0.48, 0.52, 0.47]
    treatment = [0.85, 0.88, 0.82, 0.90, 0.86]

    res = compute_statistical_comparison(baseline, treatment)
    assert res["mean_delta"] > 0.30
    assert res["is_significant_p05"] is True
    assert res["is_significant_p01"] is True
    assert res["p_value"] < 0.01
    assert res["cohens_d"] > 2.0


def test_native_rouge_l():
    ref = "The quick brown fox jumps over the lazy dog."
    cand = "The brown fox jumps over the dog."
    score = compute_rouge_l_native(ref, cand)
    assert 0.60 <= score <= 1.0

    # Disjoint text
    assert compute_rouge_l_native("Apple banana orange", "quantum mechanics physics") == 0.0


def test_error_taxonomy_classification():
    # Intrinsic contradiction
    context = "The study observed that circulation speed slowed down significantly by 15%."
    contradictory_claim = "The circulation speed increased significantly across all metrics."
    err_type = HallucinationClassifier.classify_unsupported_claim(contradictory_claim, context)
    assert err_type == "intrinsic_contradiction"

    # Entity fabulation
    fabulated_claim = "The experiment was funded by SpaceX and NASA in year 2045."
    err_type_ent = HallucinationClassifier.classify_unsupported_claim(fabulated_claim, context)
    assert err_type_ent == "entity_fabulation"

    # Refusal failure on irrelevant context
    irrelevant_context = "Photosynthesis occurs in chloroplasts of plant cells."
    err_type_refusal = HallucinationClassifier.classify_unsupported_claim(
        claim="The inflation rate in 1980 was 13.5%.",
        context=irrelevant_context,
        context_precision=0.10,
    )
    assert err_type_refusal == "refusal_failure"


def test_claim_breakdown_three_way_nli():
    evaluator = get_evaluator()
    context = [
        "James Webb Space Telescope was launched in December 2021. It orbits the Sun at Lagrange Point 2."
    ]
    # Claim 1 entailed, Claim 2 contradicts launch date, Claim 3 is neutral ungrounded speculation
    answer = (
        "James Webb Space Telescope was launched in December 2021. "
        "The telescope was launched in 1990 by Apollo astronauts. "
        "Scientists predict it will discover alien civilizations by next Tuesday."
    )

    breakdown = evaluator.compute_claim_breakdown(answer, context)
    assert breakdown["total_claims"] == 3
    assert breakdown["supported_claim_rate"] > 0.0
    assert breakdown["contradiction_rate"] > 0.0
    assert breakdown["unsupported_claim_rate"] > 0.0
    # Sum of the 3 rates must equal 1.0 (subject to floating precision)
    total_rate = breakdown["supported_claim_rate"] + breakdown["contradiction_rate"] + breakdown["unsupported_claim_rate"]
    assert round(total_rate, 2) == 1.0


def test_zero_claim_safety():
    evaluator = get_evaluator()
    # Empty string
    empty_res = evaluator.compute_claim_breakdown("", ["Some valid context"])
    assert empty_res["total_claims"] == 0
    assert empty_res["faithfulness"] == 0.0

    # Non-empty with whitespace
    ws_res = evaluator.compute_claim_breakdown("   ", ["Some valid context"])
    assert ws_res["total_claims"] == 0

    # Honest abstention
    abstain_res = evaluator.compute_claim_breakdown(
        "I cannot find sufficient information in the provided documents to answer.",
        ["Unrelated context"],
    )
    assert abstain_res["faithfulness"] == 1.0
    assert abstain_res["supported_claim_rate"] == 1.0
    assert abstain_res["contradiction_rate"] == 0.0


def test_rank_sensitive_context_precision_and_ranking():
    evaluator = get_evaluator()
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
    retrieved_ids = ["c1", "c2", "c3", "c4"]
    # Relevant chunks placed at top ranks (rank 1 and rank 2)
    top_relevant_ids = ["c1", "c2"]

    prec_top, mrr_top, ndcg_top = evaluator.compute_rank_sensitive_context_precision(
        chunks=chunks,
        question="Query",
        retrieved_ids=retrieved_ids,
        relevant_ids=top_relevant_ids,
    )
    assert prec_top == 1.0
    assert mrr_top == 1.0
    assert ndcg_top > 0.90

    # Relevant chunk placed only at lowest rank (rank 4)
    bottom_relevant_ids = ["c4"]
    prec_bot, mrr_bot, ndcg_bot = evaluator.compute_rank_sensitive_context_precision(
        chunks=chunks,
        question="Query",
        retrieved_ids=retrieved_ids,
        relevant_ids=bottom_relevant_ids,
    )
    assert prec_bot < prec_top
    assert mrr_bot == 0.25  # 1 / 4
    assert ndcg_bot < ndcg_top


def test_factual_correctness_numerical_mutation():
    evaluator = get_evaluator()
    reference = "The global average temperature increased by 1.1 degrees Celsius above pre-industrial levels."
    
    # Grounded answer
    grounded_answer = "Global average temperatures increased by 1.1 degrees Celsius above pre-industrial benchmarks."
    grounded_scores = evaluator.compute_answer_correctness(grounded_answer, reference)
    assert grounded_scores["factual_correctness"] >= 0.90

    # Mutated number (retains high lexical overlap but fact is false: 7.8 vs 1.1)
    mutated_answer = "The global average temperature increased by 7.8 degrees Celsius above pre-industrial levels."
    mutated_scores = evaluator.compute_answer_correctness(mutated_answer, reference)
    # ROUGE-L is high because 11 of 12 words match!
    assert mutated_scores["rouge_l_f1"] > 0.80
    # But factual correctness must catch the numerical mutation!
    assert mutated_scores["factual_correctness"] < grounded_scores["factual_correctness"]
    assert mutated_scores["answer_correctness"] < grounded_scores["answer_correctness"]
