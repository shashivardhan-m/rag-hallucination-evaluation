# Experiment 2: Chunk-Size Comparison Table

| Config | Chunk Size | Overlap | Context Precision (95% CI) | Context Recall (95% CI) | Faithfulness (95% CI) | Hallucination Rate | Correctness (95% CI) |
|---|---|---|---|---|---|---|---|
| **A** | 128 | 32 | **0.884 +/- 0.016** | 0.692 +/- 0.024 | 0.814 +/- 0.021 | 0.186 | 0.758 +/- 0.018 |
| **B** | **256** | 50 | 0.842 +/- 0.019 | 0.826 +/- 0.019 | **0.889 +/- 0.014** | **0.111** | **0.841 +/- 0.013** |
| **C** | 512 | 100 | 0.771 +/- 0.022 | **0.894 +/- 0.015** | 0.852 +/- 0.017 | 0.148 | 0.812 +/- 0.016 |
| **D** | 1024 | 200 | 0.648 +/- 0.026 | 0.921 +/- 0.013 | 0.791 +/- 0.023 | 0.209 | 0.749 +/- 0.021 |

*Key Empirical Finding*: 256-token chunking achieves the optimal Pareto frontier between context precision and recall, reducing hallucination rate by 46.9% relative to 1024-token chunks ($p < 0.01$).
