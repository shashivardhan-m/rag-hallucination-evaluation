# Literature Review & Theoretical Foundations

## 1. Foundations of Retrieval-Augmented Generation
Lewis et al. (NeurIPS 2020) formulated Retrieval-Augmented Generation (RAG) by combining pre-trained parametric sequence-to-sequence models with non-parametric dense vector indices (DPR, Karpukhin et al., 2020). While RAG substantially improved factual recall on open-domain QA tasks (Natural Questions, TriviaQA), subsequent empirical surveys (Ji et al., ACM Computing Surveys 2023) highlighted that neural text generators frequently disregard retrieved evidence in favor of memorized parametric priors.

## 2. Hallucination Benchmarks in RAG
- **RAGTruth (ACL 2024)**: Wu et al. introduced RAGTruth, an extensively annotated corpus of RAG hallucinations spanning multiple domains (biomedical, news, QA) and generator model families. RAGTruth establishes that over 38% of generation errors stem from subtle extrinsic extrapolation or intrinsic contradiction rather than gross factual fabrication.
- **HaluEval (EMNLP 2023)**: Li et al. constructed HaluEval, providing 35,000+ generated responses with controlled hallucination insertions. Their findings demonstrate that automated evaluation using standard similarity metrics (ROUGE, BLEU) fails to correlate with human factual assessments.
- **FActScore (EMNLP 2023)**: Min et al. introduced atomic claim decomposition, proving that breaking generation outputs into atomic subject-predicate-object propositions and verifying each individually yields significantly higher correlation with human ground truth than full-sequence similarity.

## 3. Evaluation Frameworks & Metric Formulations
- **RAGAS (Es et al., 2023)**: RAGAS formulated reference-free evaluation metrics including Faithfulness (NLI-based grounding), Answer Relevance (query-generation embedding similarity), and Context Precision (rank-weighted precision of retrieved chunks). Our work adopts and refines the RAGAS rank-sensitive Context Precision@K formulation.
- **TruLens & ARES (Saad-Falcon et al., 2024)**: Emphasized prediction confidence intervals and statistical testing in RAG evaluations to combat evaluation variance across stochastic LLM generation passes.

## 4. Cross-Encoder Reranking and Chunking Dynamics
Recent studies in neural information retrieval (Nogueira & Cho, 2019; Pradeep et al., 2021) demonstrated that dual-encoder bi-encoders suffer from information bottlenecking due to independent encoding of queries and passages. Cross-Encoders eliminate this bottleneck via full bidirectional all-to-all cross-attention across concatenated query-passage tokens. However, the exact trade-off between cross-encoder inference latency and downstream hallucination reduction has remained insufficiently characterized under controlled benchmark conditions.
