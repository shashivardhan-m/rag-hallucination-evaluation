"""Metric calculation, clamping, aggregation, and reporting helpers."""

from typing import Dict, List


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


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """Format dictionary of metrics into a clean markdown table string."""
    if not metrics:
        return "No metrics available."

    lines = [
        "| Metric Name | Value | Interpretation |",
        "| :--- | :--- | :--- |",
    ]

    interpretations = {
        "faithfulness": "Context grounding (higher is less hallucinated)",
        "hallucination_rate": "Fraction of claims unsupported by context (lower is better)",
        "answer_relevance": "Directness and relevance to the user question",
        "context_precision": "Average relevance of retrieved context to query",
        "retrieval_precision": "Fraction of retrieved chunks exceeding relevance threshold",
        "context_recall": "Coverage of reference answer content by context",
        "retrieval_recall": "Fraction of ground-truth document chunks retrieved",
        "answer_correctness": "Combined semantic (BERTScore) & lexical (ROUGE) match",
        "rouge_l_f1": "Longest common subsequence overlap with reference",
        "bertscore_f1": "Semantic token embedding similarity with reference",
    }

    for name, val in sorted(metrics.items()):
        interp = interpretations.get(name, "Evaluation metric")
        percentage = f"{val * 100:.1f}%" if 0.0 <= val <= 1.0 else f"{val:.4f}"
        lines.append(f"| `{name}` | **{val:.4f}** ({percentage}) | {interp} |")

    return "\n".join(lines)
