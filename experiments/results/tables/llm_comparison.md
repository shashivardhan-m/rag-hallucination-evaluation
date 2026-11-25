# Experiment 3: LLM Comparison Table

| Model Architecture | Parameters | Faithfulness (95% CI) | Hallucination Rate | Contradiction Rate | Unsupported Rate | Correctness (95% CI) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **LLaMA-3-8B-Instruct** | 8B | 0.884 +/- 0.016 | 0.116 | 0.031 +/- 0.005 | 0.085 +/- 0.012 | 0.832 +/- 0.014 | 480 +/- 35 |
| **Mixtral-8x7B-Instruct** | 46.7B MoE | 0.898 +/- 0.013 | 0.102 | 0.024 +/- 0.004 | 0.078 +/- 0.010 | 0.849 +/- 0.012 | 620 +/- 42 |
| **GPT-4o-mini** | Proprietary | **0.926 +/- 0.011** | **0.074** | **0.018 +/- 0.003** | **0.056 +/- 0.008** | **0.878 +/- 0.011** | 510 +/- 30 |

*Behavior with Identical Context*: When conditioned on identical retrieved chunks, the majority of hallucinations across all models are *unsupported neutral assertions* rather than *direct contradictions*, with larger parameter scales reducing extrinsic extrapolations.
