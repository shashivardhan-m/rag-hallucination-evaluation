# Research Methodology & Evaluation Framework

## 1. Experimental Benchmarks
All empirical evaluations are conducted on standardized benchmark corpora with ground-truth segment annotations:
- **RAGTruth Corpus**: Hallucination benchmark containing annotated QA pairs across biomedical, news, and technical domains with character-level hallucination labels.
- **HaluEval Benchmark**: Curated dataset comprising 10,000+ positive and perturbed hallucinated question-answer pairs designed to stress-test NLI and claim verification.

## 2. Evaluation Metrics

### 2.1 3-Way NLI Claim Breakdown
Answers are decomposed into discrete sentential claims via sentence boundary tokenization. Each claim is classified using Natural Language Inference (`cross-encoder/nli-deberta-v3-small` / standard MNLI):
- **Supported Claim Rate ($R_{\text{supp}}$)**: Proportion of claims with NLI probability concentrated on `Entailment`.
- **Contradiction Rate ($R_{\text{contra}}$)**: Proportion of claims with NLI probability concentrated on `Contradiction`.
- **Unsupported Claim Rate ($R_{\text{unsup}}$)**: Proportion of claims with NLI probability concentrated on `Neutral`.
- **Faithfulness**: Equal to $R_{\text{supp}}$. Handled safely for zero-claim outputs ($\text{len}(\text{claims}) = 0 \implies F = 1.0$) to avoid division by zero.

### 2.2 Rank-Sensitive Retrieval Metrics
Rather than unranked cosine query-chunk similarity, retrieval quality is measured using classical IR formulation (aligned with RAGAS):
- **Context Precision@K**:
  $$\text{Context Precision@K} = \frac{1}{\min(K, \text{total\_relevant})} \sum_{i=1}^K (\text{Precision@}i \cdot v_i)$$
  where $v_i \in \{0, 1\}$ indicates chunk relevance at rank $i$.
- **Mean Reciprocal Rank (MRR@K)**: Reciprocal rank of first relevant chunk.
- **Normalized Discounted Cumulative Gain (nDCG@K)**.
- **Recall@K**: Proportion of total relevant document chunks retrieved in top-$K$.

### 2.3 Answer Correctness Formulation
To avoid treating lexical overlap (ROUGE-L) or dense embedding similarity (BERTScore) as proof of factual correctness:
$$\text{Answer Correctness} = 0.20 \cdot \text{ROUGE-L} + 0.30 \cdot \text{BERTScore} + 0.50 \cdot \text{Factual Correctness}$$
where **Factual Correctness** explicitly verifies claim-level entailment against reference statements and enforces strict numerical and named-entity consistency checks.

## 3. Statistical Rigor
- **Repetitions**: Every experimental configuration is executed across $N = 5$ independent runs with controlled random seeds.
- **Confidence Intervals**: 95% Confidence Intervals computed using Student's $t$-distribution ($df = N - 1$).
- **Significance Testing**: Two-tailed paired Student's $t$-tests ($p < 0.05$, $p < 0.01$).
- **Effect Size**: Standardized Cohen's $d$ calculated across treatment vs. baseline distributions.
