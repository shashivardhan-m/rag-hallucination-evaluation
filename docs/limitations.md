# System Limitations & Future Directions

## 1. Known Architectural Limitations

### 1.1 Computational Latency of Cross-Encoder Reranking
While Cross-Encoder reranking provides the highest MRR@10 and Faithfulness gains, full cross-attention across query-document pairs scales as $\mathcal{O}(K \cdot L^2)$ with candidate pool size $K$ and sequence length $L$. On production environments with strict latency SLAs (< 100 ms), full cross-encoder passes may require distillation, pruning, or approximate early exit.

### 1.2 Boundary Artifacts in Sentence Splitting
The 3-Way NLI evaluation relies on sentential claim decomposition. In cases where an LLM generates complex clauses with nested subordinate conjunctions ("although X happened in 1990, Y was founded in 2020"), splitting at punctuation boundaries may split anaphoric references or conditional qualifications, creating isolated claims that appear unentailed when severed from their preceding clause.

### 1.3 NLI Model Biases
Pre-trained NLI models (`cross-encoder/nli-deberta-v3-small`) can exhibit subtle domain-specific biases, particularly in dense biomedical or legal jargon where medical acronyms or legal statute numbers may be misclassified as neutral when lexical overlap is low.

### 1.4 Single-Turn Focus
The current evaluation harness evaluates single-turn question-answering interactions. Multi-turn conversational RAG, where hallucinations propagate recursively across dialogue turns through coreference drift, remains an active research challenge.

## 2. Mitigations Implemented in this Project
- Native zero-dependency fallbacks for all embedding, BM25, and vector indexing operations.
- Safe division guards on all rate calculations ($\text{len}(\text{claims}) = 0 \implies F = 1.0$) preventing runtime crashes.
- Tripartite metric decoupling: Lexical overlap (ROUGE-L), semantic similarity (BERTScore), and claim-level factual entailment (Factual Correctness) are independently measured and reported.
