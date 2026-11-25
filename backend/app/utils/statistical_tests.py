"""Statistical testing, cluster-bootstrap resampling, and multiple testing corrections.

Implements rigorous statistical methods for RAG hallucination evaluation:
- Question-level clustered bootstrap confidence intervals
- McNemar's paired test for binary faithfulness outcomes (with exact binomial fallback)
- Holm-Bonferroni step-down FWER correction across retrieval comparisons
- Cohen's Kappa for NLI judge agreement against human ground truth
- Reporting helpers enforcing 2-3 decimal precision
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


def format_stat(value: float, as_percent: bool = False, decimals: int = 3) -> str:
    """Format floating point values to 2-3 decimals as required by scientific standards."""
    if value is None or np.isnan(value):
        return "N/A"
    if as_percent:
        return f"{value * 100:.1f}%"
    return f"{value:.{decimals}f}"


def cohens_kappa(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, Any]:
    """
    Computes Cohen's Kappa coefficient (κ) for inter-rater agreement between
    human ground truth and automated judge (NLI or LLM).
    Labels must be binary (0 = Hallucinated, 1 = Faithful).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have identical length")

    n = len(y_true_arr)
    if n == 0:
        return {"kappa": 0.0, "observed_agreement": 0.0, "chance_agreement": 0.0}

    # Confusion matrix elements
    # Row: true, Col: pred
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t, p] += 1

    p_observed = (cm[0, 0] + cm[1, 1]) / n

    p_true_1 = np.sum(cm[1, :]) / n
    p_true_0 = np.sum(cm[0, :]) / n
    p_pred_1 = np.sum(cm[:, 1]) / n
    p_pred_0 = np.sum(cm[:, 0]) / n

    p_expected = (p_true_1 * p_pred_1) + (p_true_0 * p_pred_0)

    if p_expected == 1.0:
        kappa = 1.0
    else:
        kappa = (p_observed - p_expected) / (1.0 - p_expected)

    # Standard error of kappa for confidence interval
    # Fleiss et al. large sample standard error
    se = np.sqrt((p_observed * (1 - p_observed)) / (n * (1 - p_expected) ** 2)) if p_expected < 1.0 else 0.0
    ci_low = max(-1.0, kappa - 1.96 * se)
    ci_high = min(1.0, kappa + 1.96 * se)

    # Landis & Koch interpretation
    if kappa > 0.80:
        interp = "Almost Perfect Agreement"
    elif kappa > 0.60:
        interp = "Substantial Agreement"
    elif kappa > 0.40:
        interp = "Moderate Agreement"
    elif kappa > 0.20:
        interp = "Fair Agreement"
    elif kappa > 0.0:
        interp = "Slight Agreement"
    else:
        interp = "Poor Agreement"

    return {
        "n": int(n),
        "kappa": round(float(kappa), 3),
        "kappa_95ci": [round(float(ci_low), 3), round(float(ci_high), 3)],
        "observed_agreement": round(float(p_observed), 3),
        "chance_agreement": round(float(p_expected), 3),
        "standard_error": round(float(se), 3),
        "interpretation": interp,
        "contingency_matrix": cm.tolist(),
    }


def mcnemar_test(
    y_true: List[int],
    y_baseline: List[int],
    y_treatment: List[int],
) -> Dict[str, Any]:
    """
    McNemar's test for paired nominal data.
    Compares binary faithfulness outcomes between two systems evaluated on the identical questions:
    - y_baseline: e.g. Unaugmented LLM (1 = Faithful, 0 = Hallucinated)
    - y_treatment: e.g. RAG System (1 = Faithful, 0 = Hallucinated)
    - y_true: optional reference (if provided, compares accuracy; if not, compares success rates directly)
    """
    a_success = np.asarray(y_baseline) == 1
    b_success = np.asarray(y_treatment) == 1

    n = len(a_success)
    if n != len(b_success):
        raise ValueError("Baseline and treatment must have identical sample length")

    # Contingency Table:
    #                 Treatment Success (1)   Treatment Fail (0)
    # Baseline (1)          a                       b (Baseline wins)
    # Baseline (0)          c (Treatment wins)      d
    a = int(np.sum(a_success & b_success))
    b = int(np.sum(a_success & ~b_success))
    c = int(np.sum(~a_success & b_success))
    d = int(np.sum(~a_success & ~b_success))

    discordant = b + c
    if discordant == 0:
        return {
            "n": n,
            "b_baseline_only": b,
            "c_treatment_only": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "significant_at_01": False,
            "method": "Exact (zero discordant pairs)",
            "contingency_table": {"both_success": a, "baseline_only": b, "treatment_only": c, "both_fail": d},
        }

    # If discordant pairs are small (<25), use exact two-sided binomial test
    if discordant < 25:
        # Under null hypothesis, b ~ Binomial(b + c, 0.5)
        # Two-sided p-value
        k = min(b, c)
        p_val = float(2.0 * stats.binom.cdf(k, discordant, 0.5))
        p_val = min(1.0, p_val)
        stat = float(b - c)
        method = "Exact Binomial Test (discordant < 25)"
    else:
        # Edwards' continuity-corrected chi-squared test
        stat = float(((abs(b - c) - 1.0) ** 2) / discordant)
        p_val = float(stats.chi2.sf(stat, df=1))
        method = "Edwards' Continuity-Corrected Chi-Squared (df=1)"

    return {
        "n": n,
        "contingency_table": {
            "both_faithful": a,
            "baseline_faithful_rag_hallucinated": b,
            "rag_faithful_baseline_hallucinated": c,
            "both_hallucinated": d,
        },
        "b_baseline_only": b,
        "c_treatment_only": c,
        "statistic": round(stat, 3),
        "p_value": round(p_val, 4),
        "significant_at_05": bool(p_val < 0.05),
        "significant_at_01": bool(p_val < 0.01),
        "method": method,
    }


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    Ensures statistical validity when testing multiple retrieval configurations (e.g. K=1, 3, 5, 10).
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    reject = [False] * m

    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = rank + 1
        adj_p = (m - k + 1) * p
        adj_p = min(1.0, max(cum_max, adj_p))
        cum_max = adj_p
        adjusted[orig_idx] = round(float(adj_p), 4)

    # Determine rejection: in Holm's method, reject as long as p_(k) <= alpha / (m - k + 1)
    all_prev_rejected = True
    results = []
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = rank + 1
        threshold = alpha / (m - k + 1)
        if all_prev_rejected and p <= threshold:
            is_rej = True
        else:
            is_rej = False
            all_prev_rejected = False
        reject[orig_idx] = is_rej

    for i in range(m):
        results.append({
            "original_index": i,
            "raw_p_value": round(float(p_values[i]), 4),
            "adjusted_p_value": adjusted[i],
            "significant": reject[i],
        })

    return results


def question_clustered_bootstrap(
    question_ids: List[Any],
    metric_values: List[float],
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Question-Level Clustered Non-Parametric Bootstrap.
    Resamples questions as independent clusters with replacement.
    Prevents artificially narrow confidence intervals caused by treating
    multiple claims from the same question as independent observations.
    """
    rng = np.random.RandomState(seed)

    # Group metric values by question ID
    q_map: Dict[Any, List[float]] = {}
    for q_id, val in zip(question_ids, metric_values):
        if val is not None and not np.isnan(val):
            q_map.setdefault(q_id, []).append(float(val))

    unique_questions = list(q_map.keys())
    n_questions = len(unique_questions)

    if n_questions == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_questions": 0}

    # Original clustered mean (mean of question means)
    q_means = np.array([np.mean(q_map[q]) for q in unique_questions])
    orig_mean = float(np.mean(q_means))

    bootstrap_means = []
    for _ in range(n_bootstraps):
        sampled_q_indices = rng.randint(0, n_questions, size=n_questions)
        sampled_means = q_means[sampled_q_indices]
        bootstrap_means.append(np.mean(sampled_means))

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_means, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1.0 - alpha / 2.0) * 100))
    std_err = float(np.std(bootstrap_means))

    return {
        "n_questions": int(n_questions),
        "mean": round(orig_mean, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "standard_error": round(std_err, 3),
        "formatted_95ci": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
        "formatted_percent_95ci": f"[{ci_lower * 100:.1f}%, {ci_upper * 100:.1f}%]",
    }
