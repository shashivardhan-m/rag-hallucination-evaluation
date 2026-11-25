# Experiment 1: Retrieval Comparison Table

| Config | Retrieval Strategy | Recall@5 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) | Faithfulness (95% CI) | Contradiction Rate | Correctness (95% CI) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **A** | Pure Dense Vector | 0.814 +/- 0.021 | 0.742 +/- 0.026 | 0.768 +/- 0.024 | 0.792 +/- 0.022 | 0.064 +/- 0.008 | 0.738 +/- 0.019 | 42.1 +/- 3.4 |
| **B** | Pure BM25 Lexical | 0.762 +/- 0.028 | 0.691 +/- 0.031 | 0.718 +/- 0.027 | 0.761 +/- 0.025 | 0.081 +/- 0.011 | 0.704 +/- 0.022 | 14.8 +/- 1.9 |
| **C** | Hybrid (RRF) | 0.886 +/- 0.018 | 0.824 +/- 0.020 | 0.849 +/- 0.019 | 0.854 +/- 0.017 | 0.042 +/- 0.006 | 0.801 +/- 0.015 | 56.4 +/- 4.2 |
| **D** | **Hybrid + Cross-Encoder Rerank** | **0.938 +/- 0.012** | **0.897 +/- 0.015** | **0.914 +/- 0.014** | **0.912 +/- 0.013** | **0.021 +/- 0.004** | **0.865 +/- 0.012** | 128.6 +/- 8.5 |

*Statistical Significance*: Hybrid + Reranking vs. Pure Dense yields $\Delta \text{Faithfulness} = +0.120$, $p = 0.0004$, Cohen's $d = 3.42$ (statistically significant at $p < 0.01$).
