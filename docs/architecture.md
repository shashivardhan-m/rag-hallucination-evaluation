# System Architecture

## Architecture Overview

```
                      +-----------------------------+
                      |     React + TypeScript      |
                      |   Interactive Web Portal    |
                      +--------------+--------------+
                                     | (REST API)
                                     v
                      +-----------------------------+
                      |       FastAPI Backend       |
                      |   Application Routing Layer |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+------------------+                                   +-------------------+
|  Document Engine |                                   | Modular Retriever |
|  Chunker (token  |                                   | - Dense (BGE/FAISS)|
|   estimation)    |                                   | - Sparse (BM25)   |
|  Vector Store    |                                   | - Hybrid (RRF)    |
|   (FAISS/NumPy)  |                                   | - Reranker (Cross)|
+--------+---------+                                   +---------+---------+
         |                                                       |
         +---------------------------+---------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   RAG Pipeline Controller   |
                      | Context Window Conditioning |
                      |    Abstention Guardrails    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     LLM Gateway Router      |
                      | (Groq, OpenAI, Anthropic,   |
                      |   Local Transformers)       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Hallucination Engine     |
                      | - 3-Way NLI Claim Break     |
                      | - Rank-Sensitive IR Metrics |
                      | - Factual Claim Verification|
                      | - 5-Class Error Taxonomy    |
                      +-----------------------------+
```

## Subsystem Responsibilities

1. **Document Processing (`app.core.document_processor`)**:
   - Ingests PDF, DOCX, TXT files.
   - Cleans and normalizes Unicode strings.
   - Chunks text into user-specified token windows (128, 256, 512, 1024) with sliding overlap.

2. **Embedding and Vector Index (`app.core.embeddings`, `app.core.vector_store`)**:
   - Generates dense normalized embeddings via `BAAI/bge-small-en-v1.5` (or fallback).
   - Manages FAISS IndexFlatIP inner-product search for exact cosine similarity ranking.

3. **Multi-Strategy Retrieval (`app.core.retriever`)**:
   - `retrieve_semantic`: Dense inner-product vector search.
   - `retrieve_bm25`: Lexical BM25Okapi scoring with Lucene-style IDF and length normalization.
   - `retrieve_hybrid`: Reciprocal Rank Fusion / weighted linear score combination.
   - `retrieve_reranked`: 2-stage retrieval feeding candidate pool to `cross-encoder/ms-marco-MiniLM-L-6-v2`.

4. **Hallucination Evaluation & Taxonomy (`app.core.evaluator`)**:
   - Claim decomposition via sentence boundary tokenization.
   - 3-Way NLI classification: Supported (Entailment), Contradiction, Unsupported (Neutral).
   - Error categorization into 5 academic error classes: Intrinsic Contradiction, Extrinsic Extrapolation, Entity Fabulation, Cross-Chunk Confusion, Refusal Failure.
   - Reference-based factual verification decoupled from lexical ROUGE-L and embedding BERTScore.
