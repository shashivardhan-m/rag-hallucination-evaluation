"""Statistical metric calculation, confidence intervals, hypothesis testing, and reporting helpers."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero or invalid."""
    try:
        if denominator == 0 or denominator is None or numerator is None:
            return float(default)
        return float(numerator / denominator)
    except Exception:
        return float(default)


def clamp_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a floating point score between min_val and max_val, handling NaN/inf."""
    if score is None:
        return min_val
    try:
        val = float(score)
        if val != val:  # NaN check
            return min_val
        return max(min_val, min(max_val, val))
    except (TypeError, ValueError):
        return min_val


def compute_rouge_l_native(reference: str, candidate: str) -> float:
    """
    Pure Python longest common subsequence (LCS) ROUGE-L F1 calculation.
    Zero-dependency fallback when rouge-score package is unavailable.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    # DP table for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    prec = lcs_len / n
    rec = lcs_len / m
    if prec + rec == 0:
        return 0.0
    f1 = (2 * prec * rec) / (prec + rec)
    return float(f1)


# ----------------------------------------------------------------------
# Statistical Significance & Confidence Intervals
# ----------------------------------------------------------------------

def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute sample mean, standard deviation, standard error, and 95% Confidence Interval.
    Uses Student's t distribution when N < 30 and Normal approximation when N >= 30.
    """
    clean_vals = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    n = len(clean_vals)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci_margin": 0.0,
            "n": 0,
        }

    mean = float(np.mean(clean_vals))
    if n == 1:
        return {
            "mean": round(mean, 4),
            "std": 0.0,
            "se": 0.0,
            "ci_low": round(mean, 4),
            "ci_high": round(mean, 4),
            "ci_margin": 0.0,
            "n": 1,
        }

    std = float(np.std(clean_vals, ddof=1))
    se = std / math.sqrt(n)

    # Critical value from Student's t or Normal distribution
    alpha = 1.0 - confidence
    if n < 30:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    else:
        t_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    margin = t_crit * se
    ci_low = max(0.0, mean - margin)
    ci_high = min(1.0, mean + margin) if mean <= 1.0 else (mean + margin)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "se": round(se, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "ci_margin": round(margin, 4),
        "n": n,
    }


def compute_bootstrap_ci(
    values: List[float],
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Non-parametric bootstrap confidence interval estimation with replacement."""
    clean_vals = np.array([float(v) for v in values if isinstance(v, (int, float)) and v == v])
    n = len(clean_vals)
    if n < 2:
        val = float(clean_vals[0]) if n == 1 else 0.0
        return {"mean": val, "ci_low": val, "ci_high": val, "n": n}

    rng = np.random.default_rng(random_seed)
    boot_means = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        resample = rng.choice(clean_vals, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    alpha = 1.0 - confidence
    low_pct = (alpha / 2.0) * 100
    high_pct = (1.0 - alpha / 2.0) * 100

    ci_low = float(np.percentile(boot_means, low_pct))
    ci_high = float(np.percentile(boot_means, high_pct))
    mean = float(np.mean(clean_vals))

    return {
        "mean": round(mean, 4),
        "ci_low": round(max(0.0, ci_low), 4),
        "ci_high": round(ci_high, 4),
        "n": n,
    }


def compute_statistical_comparison(
    baseline_values: List[float],
    treatment_values: List[float],
) -> Dict[str, Any]:
    """
    Perform paired two-tailed Student's t-test and effect size (Cohen's d)
    contrasting a treatment condition (e.g. Hybrid RAG with reranking) against a baseline.
    """
    clean_base = [float(v) for v in baseline_values if isinstance(v, (int, float)) and v == v]
    clean_treat = [float(v) for v in treatment_values if isinstance(v, (int, float)) and v == v]

    min_len = min(len(clean_base), len(clean_treat))
    if min_len < 2:
        return {
            "p_value": 1.0,
            "t_statistic": 0.0,
            "is_significant_p05": False,
            "is_significant_p01": False,
            "cohens_d": 0.0,
            "mean_delta": 0.0,
        }

    b = np.array(clean_base[:min_len])
    t = np.array(clean_treat[:min_len])
    diff = t - b
    mean_delta = float(np.mean(diff))

    # Paired t-test
    if np.all(diff == 0):
        t_stat = 0.0
        p_val = 1.0
    else:
        res = stats.ttest_rel(t, b)
        t_stat = float(res.statistic) if res.statistic == res.statistic else 0.0
        p_val = float(res.pvalue) if res.pvalue == res.pvalue else 1.0

    # Cohen's d for paired samples
    s_diff = float(np.std(diff, ddof=1))
    cohens_d = (mean_delta / s_diff) if s_diff > 1e-9 else 0.0

    return {
        "p_value": round(p_val, 6),
        "t_statistic": round(t_stat, 4),
        "is_significant_p05": bool(p_val < 0.05),
        "is_significant_p01": bool(p_val < 0.01),
        "cohens_d": round(cohens_d, 4),
        "mean_delta": round(mean_delta, 4),
    }


def aggregate_metrics(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate mean of metric values across a list of evaluation metric dictionaries."""
    if not metric_dicts:
        return {}

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for m in metric_dicts:
        for k, v in m.items():
            if isinstance(v, (int, float)) and v == v:  # not NaN
                totals[k] = totals.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

    averages: Dict[str, float] = {}
    for k, total in totals.items():
        cnt = counts.get(k, 1)
        averages[k] = round(total / cnt, 4)

    return averages


def aggregate_metrics_with_stats(
    metric_dicts: List[Dict[str, float]],
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive metric aggregation returning mean, std, standard error,
    and 95% Confidence Intervals for every evaluated metric.
    """
    if not metric_dicts:
        return {}

    metric_series: Dict[str, List[float]] = {}
    for m in metric_dicts:
        for k, v in m.items():
            if isinstance(v, (int, float)) and v == v:
                metric_series.setdefault(k, []).append(float(v))

    stats_summary: Dict[str, Dict[str, float]] = {}
    for k, vals in metric_series.items():
        stats_summary[k] = compute_confidence_interval(vals, confidence=confidence)

    return stats_summary


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format dictionary of metrics into a clean markdown table string including confidence intervals."""
    if not metrics:
        return "No metrics available."

    lines = [
        "| Metric Name | Mean (95% CI) | Std Dev | Interpretation |",
        "| :--- | :--- | :--- | :--- |",
    ]

    interpretations = {
        "faithfulness": "Context grounding (higher is less hallucinated)",
        "hallucination_rate": "Fraction of claims unsupported by context (lower is better)",
        "answer_relevance": "Directness and semantic alignment to user question",
        "context_precision": "Average semantic relevance of retrieved context to query",
        "retrieval_precision": "Fraction of retrieved chunks exceeding relevance threshold",
        "context_recall": "Coverage of reference answer content by context",
        "retrieval_recall": "Fraction of ground-truth document chunks retrieved",
        "answer_correctness": "Composite semantic (BERTScore) & lexical (ROUGE) alignment",
        "rouge_l_f1": "Longest common subsequence overlap with reference",
        "bertscore_f1": "Token-level contextual embedding similarity with reference",
    }

    for name, stat in sorted(metrics.items()):
        interp = interpretations.get(name, "Evaluation metric")
        if isinstance(stat, dict) and "mean" in stat and "ci_low" in stat:
            mean = stat["mean"]
            ci_l = stat["ci_low"]
            ci_h = stat["ci_high"]
            std = stat.get("std", 0.0)
            lines.append(
                f"| `{name}` | **{mean:.4f}** `[{ci_l:.4f}, {ci_h:.4f}]` | ±{std:.4f} | {interp} |"
            )
        elif isinstance(stat, (int, float)):
            percentage = f"{stat * 100:.1f}%" if 0.0 <= stat <= 1.0 else f"{stat:.4f}"
            lines.append(f"| `{name}` | **{stat:.4f}** ({percentage}) | N/A | {interp} |")

    return "\n".join(lines)
