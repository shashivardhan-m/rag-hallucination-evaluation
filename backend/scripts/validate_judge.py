"""Validates the NLI Faithfulness Judge against 100 hand-labeled ground-truth outputs.

Computes:
- Judge Accuracy, Precision, Recall, Macro F1
- Inter-annotator agreement using Cohen's Kappa (κ) with 95% CI
- Per-hallucination-type error breakdown
Saves detailed JSON report and prints an academic verification table.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Inject version check shim for transformers if needed
import types
if "transformers.dependency_versions_check" not in sys.modules:
    shim = types.ModuleType("transformers.dependency_versions_check")
    shim.dep_version_check = lambda *a, **k: None
    sys.modules["transformers.dependency_versions_check"] = shim

from app.core.evaluator import get_evaluator
from app.utils.statistical_tests import cohens_kappa, format_stat


def validate_nli_judge(
    dataset_path: Path,
    decision_threshold: float = 0.70,
):
    with open(dataset_path, "r") as f:
        samples = json.load(f)

    evaluator = get_evaluator()

    y_true = []
    y_pred = []
    continuous_scores = []
    detailed_results = []

    print(f"\n{'=' * 72}")
    print(f"VALIDATING NLI JUDGE ON 100 HAND-LABELED GROUND-TRUTH SAMPLES")
    print(f"Dataset: {dataset_path.name} (N = {len(samples)})")
    print(f"Decision Threshold: Faithfulness >= {decision_threshold:.2f} -> Faithful (1), else Hallucinated (0)")
    print(f"{'=' * 72}\n")

    for s in samples:
        s_id = s["id"]
        ctx = [s["context"]]
        ans = s["answer"]
        human_label = int(s["human_ground_truth"])

        faithfulness = evaluator.compute_faithfulness(answer=ans, context_chunks=ctx)
        pred_label = 1 if faithfulness >= decision_threshold else 0

        y_true.append(human_label)
        y_pred.append(pred_label)
        continuous_scores.append(faithfulness)

        detailed_results.append({
            "id": s_id,
            "domain": s.get("domain", ""),
            "human_label": human_label,
            "judge_score": round(faithfulness, 3),
            "judge_pred": pred_label,
            "is_correct": bool(human_label == pred_label),
            "hallucination_type": s.get("hallucination_type", "none"),
        })

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    n = len(y_true)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    # Confusion matrix
    # Row 0: True Hallucinated, Row 1: True Faithful
    # Col 0: Pred Hallucinated, Col 1: Pred Faithful
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))

    prec_faithful = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_faithful = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_faithful = (2 * prec_faithful * rec_faithful) / (prec_faithful + rec_faithful) if (prec_faithful + rec_faithful) > 0 else 0.0

    prec_halluc = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_halluc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_halluc = (2 * prec_halluc * rec_halluc) / (prec_halluc + rec_halluc) if (prec_halluc + rec_halluc) > 0 else 0.0

    macro_f1 = (f1_faithful + f1_halluc) / 2.0

    # Cohen's Kappa
    kappa_stats = cohens_kappa(y_true, y_pred)

    # Breakdown by hallucination type
    h_types = sorted(list(set(s.get("hallucination_type", "none") for s in samples if s.get("hallucination_type") != "none")))
    type_breakdown = {}
    for ht in h_types:
        sub = [d for d in detailed_results if d["hallucination_type"] == ht]
        correct = sum(1 for d in sub if d["judge_pred"] == 0)  # correctly caught as hallucinated
        type_breakdown[ht] = {
            "total": len(sub),
            "caught_as_hallucination": correct,
            "detection_rate": round(correct / len(sub), 3) if sub else 0.0,
        }

    report = {
        "benchmark_file": str(dataset_path),
        "sample_size": n,
        "true_faithful_count": int(np.sum(y_true_arr == 1)),
        "true_hallucinated_count": int(np.sum(y_true_arr == 0)),
        "decision_threshold": decision_threshold,
        "confusion_matrix": {
            "true_negatives_correct_hallucination": tn,
            "false_positives_missed_hallucination": fp,
            "false_negatives_falsely_accused": fn,
            "true_positives_correct_faithful": tp,
        },
        "metrics": {
            "accuracy": round(accuracy, 3),
            "macro_f1": round(macro_f1, 3),
            "cohens_kappa": kappa_stats["kappa"],
            "cohens_kappa_95ci": kappa_stats["kappa_95ci"],
            "cohens_kappa_interpretation": kappa_stats["interpretation"],
            "observed_agreement": kappa_stats["observed_agreement"],
            "chance_agreement": kappa_stats["chance_agreement"],
            "faithful_class": {
                "precision": round(prec_faithful, 3),
                "recall": round(rec_faithful, 3),
                "f1": round(f1_faithful, 3),
            },
            "hallucinated_class": {
                "precision": round(prec_halluc, 3),
                "recall": round(rec_halluc, 3),
                "f1": round(f1_halluc, 3),
            },
        },
        "hallucination_type_detection": type_breakdown,
        "detailed_evaluations": detailed_results,
    }

    # Print Formatted Verification Table
    print("### NLI Judge Validation Results (Against N=100 Ground-Truth Human Labels)\n")
    print(f"| Metric | Value | 95% Confidence Interval | Scientific Interpretation |")
    print(f"|---|---|---|---|")
    print(f"| **Overall Accuracy** | **{format_stat(accuracy, as_percent=True)}** ({tp + tn}/{n}) | — | Correctly classified 100 benchmark items |")
    print(f"| **Cohen's Kappa (κ)** | **{kappa_stats['kappa']:.3f}** | [{kappa_stats['kappa_95ci'][0]:.3f}, {kappa_stats['kappa_95ci'][1]:.3f}] | **{kappa_stats['interpretation']}** |")
    print(f"| **Observed Agreement ($p_o$)** | {format_stat(kappa_stats['observed_agreement'], as_percent=True)} | — | Empirical human-judge agreement |")
    print(f"| **Chance Agreement ($p_e$)** | {format_stat(kappa_stats['chance_agreement'], as_percent=True)} | — | Expected agreement by random guessing |")
    print(f"| **Macro F1 Score** | **{macro_f1:.3f}** | — | Balanced performance across both classes |")
    print(f"| **Hallucination Detection F1** | **{f1_halluc:.3f}** | — | Recall: {format_stat(rec_halluc, as_percent=True)}, Precision: {format_stat(prec_halluc, as_percent=True)} |")
    print(f"| **Faithfulness F1** | **{f1_faithful:.3f}** | — | Recall: {format_stat(rec_faithful, as_percent=True)}, Precision: {format_stat(prec_faithful, as_percent=True)} |")

    print("\n### Confusion Matrix Breakdown")
    print(f"  • True Positives  (Faithful correctly validated)    : {tp}")
    print(f"  • True Negatives  (Hallucination correctly caught)  : {tn}")
    print(f"  • False Positives (Hallucination missed by judge)   : {fp}")
    print(f"  • False Negatives (Faithful falsely flagged)        : {fn}")

    print("\n### Detection Rate by Hallucination Taxonomy Category")
    for ht, data in type_breakdown.items():
        print(f"  • {ht:28s}: {data['caught_as_hallucination']}/{data['total']} detected ({data['detection_rate'] * 100:.1f}%)")

    out_file = dataset_path.parent / "judge_validation_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report saved to: {out_file}")
    return report


if __name__ == "__main__":
    bench_file = Path("/Users/shashivardhanreddymallepally/Downloads/rag-hallucination-eval/backend/data/benchmarks/judge_validation_100.json")
    validate_nli_judge(bench_file)
