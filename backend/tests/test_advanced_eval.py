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
