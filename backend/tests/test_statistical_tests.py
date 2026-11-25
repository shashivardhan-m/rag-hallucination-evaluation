"""Unit tests for statistical testing, bootstrapping, and multiple comparison correction."""

import numpy as np
import pytest

from app.utils.statistical_tests import (
    cohens_kappa,
    format_stat,
    holm_bonferroni_correction,
    mcnemar_test,
    question_clustered_bootstrap,
)


def test_cohens_kappa_perfect_agreement():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 1, 0, 0, 1, 0]
    res = cohens_kappa(y_true, y_pred)
    assert res["kappa"] == 1.0
    assert res["observed_agreement"] == 1.0
    assert "Almost Perfect" in res["interpretation"]


def test_cohens_kappa_chance_agreement():
    # Symmetric 50-50 disagreement
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]
    res = cohens_kappa(y_true, y_pred)
    assert res["kappa"] == -1.0
    assert res["observed_agreement"] == 0.0


def test_mcnemar_test_identical_models():
    y_baseline = [1, 0, 1, 1, 0]
    y_treatment = [1, 0, 1, 1, 0]
    res = mcnemar_test([], y_baseline, y_treatment)
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0
    assert res["b_baseline_only"] == 0
    assert res["c_treatment_only"] == 0


def test_mcnemar_test_treatment_superiority():
    # Treatment improves 10 cases that baseline failed, baseline never beats treatment
    y_baseline = [0] * 10 + [1] * 10
    y_treatment = [1] * 10 + [1] * 10
    res = mcnemar_test([], y_baseline, y_treatment)
    assert res["c_treatment_only"] == 10
    assert res["b_baseline_only"] == 0
    assert res["p_value"] < 0.01
    assert res["significant_at_05"] is True


def test_holm_bonferroni_correction():
    p_values = [0.001, 0.015, 0.035, 0.060]
    res = holm_bonferroni_correction(p_values, alpha=0.05)
    assert len(res) == 4
    # Smallest p-value multiplied by m (4 * 0.001 = 0.004)
    assert res[0]["adjusted_p_value"] == 0.004
    assert res[0]["significant"] is True
    # Verify monotonicity of adjusted p-values
    adj_vals = [r["adjusted_p_value"] for r in res]
    assert adj_vals == sorted(adj_vals)


def test_question_clustered_bootstrap():
    # 5 distinct questions with multiple claims per question
    q_ids = ["Q1", "Q1", "Q2", "Q2", "Q3", "Q4", "Q5"]
    values = [0.8, 0.9, 0.4, 0.5, 0.7, 0.85, 0.95]

    res = question_clustered_bootstrap(q_ids, values, n_bootstraps=500, seed=42)
    assert res["n_questions"] == 5
    assert 0.0 <= res["ci_lower"] <= res["mean"] <= res["ci_upper"] <= 1.0
    assert "formatted_95ci" in res


def test_format_stat():
    assert format_stat(0.81456, decimals=3) == "0.815"
    assert format_stat(0.81456, as_percent=True, decimals=3) == "81.5%"
    assert format_stat(np.nan) == "N/A"
