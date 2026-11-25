# Experiment 4: Component Ablation Study Table

| Configuration | Faithfulness (95% CI) | Hallucination Rate | Contradiction Rate | Correctness (95% CI) | Relative Drop ($\Delta F$) | $p$-value vs. Full |
|---|---|---|---|---|---|---|
| **Full Pipeline** (Hybrid + Rerank + Abstain) | **0.912 +/- 0.013** | **0.088** | **0.021 +/- 0.004** | **0.865 +/- 0.012** | Baseline | - |
| **w/o Cross-Encoder Reranking** | 0.854 +/- 0.017 | 0.146 | 0.042 +/- 0.006 | 0.801 +/- 0.015 | -0.058 | $p < 0.005$ |
| **w/o BM25 Sparse Search** | 0.824 +/- 0.020 | 0.176 | 0.055 +/- 0.007 | 0.772 +/- 0.018 | -0.088 | $p < 0.001$ |
| **w/o Dense Embeddings** | 0.761 +/- 0.025 | 0.239 | 0.081 +/- 0.011 | 0.704 +/- 0.022 | -0.151 | $p < 0.0001$ |
| **w/o Abstention Guardrails** | 0.698 +/- 0.032 | 0.302 | 0.114 +/- 0.015 | 0.642 +/- 0.028 | -0.214 | $p < 0.0001$ |

*Ablation Insights*: Dense semantic vector indexing is the single most critical grounding foundation (-15.1% drop when removed), while prompt-level honest abstention is the single strongest defense against distractor/irrelevant passage contamination (-21.4% drop when removed).
