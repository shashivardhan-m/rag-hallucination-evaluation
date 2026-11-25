# Research Questions

This project empirically investigates five core research questions governing factual reliability in Retrieval-Augmented Generation:

### RQ1: Which retrieval strategy minimizes unsupported assertions?
> *Does dense semantic vector search, sparse BM25 lexical search, or hybrid reciprocal rank fusion achieve the highest factual grounding and information retrieval rank metrics (Recall@5, MRR@10, nDCG@10)?*

### RQ2: Does Cross-Encoder reranking improve downstream factual consistency?
> *Does adding a computationally expensive cross-encoder reranking step translate to statistically significant reductions in generation hallucinations ($p < 0.05$), or does it merely reorder already-retrieved candidates with diminishing returns?*

### RQ3: What is the optimal chunk size for balancing context grounding vs. relevance dilution?
> *How does chunk window granularity (128 vs 256 vs 512 vs 1024 tokens) influence retrieval precision, context recall, and parametric hallucinations? Does smaller chunking eliminate irrelevant noise or truncate critical semantic context?*

### RQ4: How does generator behavior degrade under corrupted or irrelevant context?
> *When retrieved context consists of distractor passages or entirely irrelevant documents, do models properly abstain (Honest Refusal), or do they succumb to Refusal Failure and hallucinate synthetic extrapolations?*

### RQ5: How do heterogeneous LLM architectures behave under identical retrieved contexts?
> *When given strictly identical retrieved passages, how do different models (e.g. LLaMA-3-8B-Instruct, Mixtral-8x7B-Instruct, Claude-3.5-Sonnet) differ in adherence to context, vulnerability to contradictory claims, and overall factual correctness?*
