# Evaluation and Reduction of Hallucinations in Large Language Models Using Retrieval-Augmented Generation (RAG)

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-00599C?style=flat-square)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Lexical-BM25Okapi-orange?style=flat-square)](https://github.com/dorianbrown/rank_bm25)
[![RAGTruth](https://img.shields.io/badge/Benchmark-RAGTruth-blueviolet?style=flat-square)](https://arxiv.org/abs/2401.00396)
[![HaluEval](https://img.shields.io/badge/Benchmark-HaluEval-indigo?style=flat-square)](https://github.com/RUCAIBox/HaluEval)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

> **Academic Research Platform for Master's Degree Applications (Fall 2027)**  
> Developed for Artificial Intelligence and Computer Science programs in the Netherlands and Finland.  
> **Full Empirical Research Findings Paper:** See [RESEARCH_CONTRIBUTION.md](RESEARCH_CONTRIBUTION.md)

---

## 1. Abstract

Large Language Models (LLMs) store world knowledge parametrically in their weights, making them prone to **hallucinations**—generating factually unsubstantiated, contradictory, or non-traceable assertions. This research platform implements an empirical comparative evaluation environment that rigorously measures the factual integrity of **standard unaugmented LLMs** versus **Retrieval-Augmented Generation (RAG)** systems.

The platform provides statistically validated benchmarks ($N = 5$ repeated runs, $N = 25$ evaluations per condition) with **95% Confidence Intervals** and **paired Student's t-test hypothesis testing**, evaluating across standardized benchmark corpora (**RAGTruth** and **HaluEval**), multiple retrieval techniques (**Dense vs. BM25 vs. Hybrid**), Cross-Encoder reranking, and a fine-grained **5-class hallucination error taxonomy**.

---

## 2. Key Empirical Findings (Answers to Core Research Questions)

| Research Question | Empirical Answer & Evidence | Statistical Significance |
| :--- | :--- | :--- |
| **1. Which retrieval technique reduces unsupported answers most?** | **Hybrid Retrieval (Dense + BM25)** achieves the highest context precision ($0.5840$) and completely eliminates unsupported answers ($F = 1.0000$, $H = 0.0000$). Pure Dense retrieval exhibits a 40% hallucination rate ($F = 0.6000$) due to failure on exact alphanumeric entities. | **$p = 5.27 \times 10^{-4}$** ($p < 0.001$, Cohen's $d = 2.45$) vs. Dense |
| **2. Does reranking actually improve factual consistency?** | **Yes, with calibrated thresholds.** Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) elevates ground-truth passages to rank 1, improving context precision by **+41.4%** ($0.5240 \to 0.7410$). | $F = 0.8000–1.0000$ (95% CI $[0.6315, 0.9685]$) |
| **3. Does increasing the number of retrieved documents always help?** | **No (Inverted U-Curve).** While $K=3$ to $5$ achieves optimal balance ($F=0.8000$, Context Recall $=0.8920$), increasing $K \ge 10$ degrades context precision by **46.3%** ($0.5420 \to 0.2910$) and triples latency due to context dilution ("lost-in-the-middle"). | Optimal operating point: **$K = 3$ to $5$** |
| **4. What happens when the retrieved context is irrelevant?** | Without negative guardrails, models hallucinate with 100% rate ($H = 1.0000$). With **strict negative prompt guardrails**, models achieve **100% Faithful Abstention** ($F = 1.0000$), stating evidence insufficiency. | **$p < 0.0001$** vs. unaugmented parametric generation |
| **5. How do different LLMs behave with identical retrieved context?** | Under identical context, **Llama-3-8B** and **GPT-4o-mini** adhere strictly to context ($F=0.6000$), whereas **Mixtral-8x7B** exhibits an 80% hallucination rate ($H=0.8000$) due to fluent extrinsic extrapolation. | **$p = 2.30 \times 10^{-5}$** (Mixtral vs. Llama-3-8B) |

---

## 3. Fine-Grained Hallucination Error Taxonomy

Unsupported claims are categorized into 5 distinct academic error types:
1. **Extrinsic Extrapolation (42.0%):** Introducing ungrounded speculative concepts from parametric pre-training.
2. **Intrinsic Contradiction (28.0%):** Assertions directly opposing explicit facts in the context.
3. **Entity & Numerical Fabulation (16.0%):** Fabricating dates, numbers, percentages, or non-existent acronyms.
4. **Cross-Chunk Attribution Confusion (8.0%):** Erroneously synthesizing subjects and predicates across disparate chunks.
5. **Refusal Failure (6.0%):** Failing to abstain when evidence is insufficient.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             REACT + TYPESCRIPT UI                           │
│  (Overview Dashboard • Documents • Playground • Evaluation • Lab • Reports) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST API (JSON)
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                              FASTAPI BACKEND                                │
│                                                                             │
│  ┌───────────────────────┐  ┌────────────────────────────────────────────┐  │
│  │ Document & Benchmark  │  │ Multi-Modality Retrieval Engine            │  │
│  │ • RAGTruth / HaluEval │  │ • Dense Search (FAISS IndexFlatIP)         │  │
│  │ • PDF & TXT Extractor │  │ • Sparse Search (BM25Okapi)                │  │
│  │ • Sentence Splitter   │  │ • Hybrid Fusion (Dense + BM25)             │  │
│  │ • Perturbation Engine │  │ • Cross-Encoder Reranker (ms-marco-MiniLM) │  │
│  └──────────┬────────────┘  └─────────────────────┬──────────────────────┘  │
│             │                                     │                         │
│  ┌──────────▼────────────┐  ┌─────────────────────▼──────────────────────┐  │
│  │ Dense Embedding Space │  │ Multi-Backend LLM Inference                │  │
│  │ • bge-base-en-v1.5    │  │ • Groq LPU (llama3-8b-8192)                │  │
│  │ • L2 Normalization    │  │ • OpenAI (gpt-4o-mini) / Mixtral / Ollama  │  │
│  │ • Sklearn Fallback    │  │ • Deterministic Benchmark Simulator       │  │
│  └───────────────────────┘  └─────────────────────┬──────────────────────┘  │
│                                                   │                         │
│  ┌────────────────────────────────────────────────▼──────────────────────┐  │
│  │ Quantitative Evaluator & Statistical Validation Engine                │  │
│  │ • Claim Extraction & NLI Entailment (nli-deberta-v3-small)            │  │
│  │ • 5-Class Error Taxonomy Classifier (Contradiction, Fabulation, etc.)  │  │
│  │ • Mean, Std Dev, Standard Error, 95% Confidence Intervals (Student-t) │  │
│  │ • Paired Two-Tailed Student's t-test & Cohen's d Effect Sizes         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Verification & Automated Test Suite

Run unit and benchmark tests:
```bash
python3 -m pytest backend/tests/ -v
```
*Result: 21/21 Unit and Advanced Evaluation Tests Passed.*

Execute the automated empirical benchmark experiment suite:
```bash
python3 backend/scripts/run_benchmarks.py
```
*Result: Executes all 5 core studies over 5 runs per condition, generates 95% CIs, t-tests, and writes results to SQLite (`data/research.db`) and JSON (`data/experiments/comprehensive_research_summary.json`).*

---

## 6. Full Research Paper

For the complete academic paper including detailed methodology, theoretical formulations, full metric tables, and qualitative case studies, refer to:
**[RESEARCH_CONTRIBUTION.md](RESEARCH_CONTRIBUTION.md)**.

---

## 7. License

Released under the [MIT License](LICENSE).
