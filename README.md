# Evaluation and Reduction of Hallucinations in Large Language Models Using Retrieval-Augmented Generation (RAG)

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-00599C?style=flat-square)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--base--en--v1.5-FFD21E?style=flat-square)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

> **Academic Research Platform for Master's Degree Applications (Fall 2027)**  
> Developed for Artificial Intelligence and Computer Science programs in the Netherlands and Finland.

---

## 1. Abstract

Large Language Models (LLMs) store world knowledge parametrically in their weights, making them prone to **hallucinations**—generating factually unsubstantiated, contradictory, or non-traceable assertions. This research platform implements a comparative evaluation environment that contrasts **standard unaugmented LLMs** against **Retrieval-Augmented Generation (RAG)** systems. 

The platform supports empirical hallucination evaluation using **Natural Language Inference (NLI)** claim-level verification (`cross-encoder/nli-deberta-v3-small`), lexical similarity (ROUGE-L), and contextual semantic embeddings (BERTScore). Researchers can perform ablation experiments across chunk sizes, dense vs. hybrid (BM25 + Dense) retrieval, and cross-encoder reranking.

---

## 2. System Architecture

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
│  │ Document Ingestion    │  │ Retrieval Engine                           │  │
│  │ • PDF & TXT Extractor │  │ • Dense Search (FAISS IndexFlatIP)         │  │
│  │ • Sentence Splitter   │  │ • Sparse Search (BM25Okapi)                │  │
│  │ • Overlapping Chunks  │  │ • Cross-Encoder Reranker (ms-marco-MiniLM) │  │
│  └──────────┬────────────┘  └─────────────────────┬──────────────────────┘  │
│             │                                     │                         │
│  ┌──────────▼────────────┐  ┌─────────────────────▼──────────────────────┐  │
│  │ Dense Embedding Space │  │ Multi-Backend LLM Inference                │  │
│  │ • bge-base-en-v1.5    │  │ • Groq LPU (llama3-8b-8192)                │  │
│  │ • L2 Normalization    │  │ • OpenAI (gpt-4o-mini) / Ollama / HF       │  │
│  └───────────────────────┘  └─────────────────────┬──────────────────────┘  │
│                                                   │                         │
│  ┌────────────────────────────────────────────────▼──────────────────────┐  │
│  │ Quantitative Hallucination Evaluator                                  │  │
│  │ • Claim Extraction & NLI Entailment (nli-deberta-v3-small)            │  │
│  │ • Faithfulness • Hallucination Rate • Answer Relevance                │  │
│  │ • Context Precision/Recall • ROUGE-L F1 • BERTScore F1                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Modules

| Module | Purpose | Implementation Details |
|---|---|---|
| **Module 1: Document Management** | Ingest, parse, and chunk research papers | `pypdf`, sentence-boundary regex, configurable chunk size ($128$–$1024$) and overlap ($0$–$256$). |
| **Module 2: Embedding & Retrieval** | Dense vector search and hybrid fusion | `BAAI/bge-base-en-v1.5` embeddings (768d), FAISS `IndexFlatIP`, BM25 keyword index, Reciprocal Rank Fusion. |
| **Module 3: Q&A Playground** | Comparative execution | Side-by-side generation: unprompted LLM vs. context-grounded RAG with evidence attribution. |
| **Module 4: Hallucination Evaluation** | Formal metric calculation | Reference-free NLI claim grounding + reference-based golden answer alignment. |
| **Module 5: Experiment Lab** | Sensitivity & ablation studies | Automated parameter grid sweeps across chunk size, top-K, and retrieval modalities. |
| **Module 6: Research Analytics** | Longitudinal visualization | KPI cards, time-series hallucination rates, radar comparison, and latency profiles. |
| **Module 7: Report Generation** | Academic dissemination | Export raw trials to CSV/JSON, dynamic markdown research report compiler. |

---

## 4. Evaluation Metrics & Theoretical Formulation

### Reference-Free Metrics (Unsupervised Grounding)

1. **Faithfulness ($F$):**
   $$\text{Faithfulness} = \frac{|\{c \in \mathcal{C} \mid \text{NLI}(c, \mathcal{K}) = \text{Entailment}\}|}{|\mathcal{C}|}$$
   Where $\mathcal{C}$ is the set of atomic claims extracted from the answer, and $\mathcal{K}$ is the concatenated retrieved context. Evaluated using `cross-encoder/nli-deberta-v3-small`.

2. **Hallucination Rate ($H$):**
   $$H = 1.0 - \text{Faithfulness}$$

3. **Answer Relevance ($R$):**
   $$R = \cos(\mathbf{e}_Q, \mathbf{e}_A) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_A}{\|\mathbf{e}_Q\| \|\mathbf{e}_A\|}$$
   Evaluates whether the generation directly satisfies the user prompt semantically.

4. **Context Precision ($P_{\text{ctx}}$):**
   $$P_{\text{ctx}} = \frac{1}{K} \sum_{k=1}^K \cos(\mathbf{e}_Q, \mathbf{e}_{c_k})$$

### Reference-Based Metrics (Supervised Benchmark)

1. **Answer Correctness ($S_{\text{corr}}$):**
   $$S_{\text{corr}} = 0.4 \times \text{ROUGE-L}_{F_1} + 0.6 \times \text{BERTScore}_{F_1}$$

2. **Context Recall ($R_{\text{ctx}}$):**
   Semantic alignment between the ground truth golden reference and the retrieved context chunks.

---

## 5. Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ & npm
- Free [Groq API Key](https://console.groq.com) (recommended default LLM provider)

### One-Command Launch

```bash
git clone <repo-url>
cd rag-hallucination-eval
./start.sh
```

The script automatically prepares Python `venv`, installs frontend dependencies, starts the backend on `http://localhost:8000`, and launches the Vite server on `http://localhost:5173`.

### Manual Setup

#### 1. Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set your GROQ_API_KEY (optional, fallback offline models included)

# Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:5173** to access the research dashboard.

---

## 6. Verification & Automated Test Suite

### Unit Testing
Run the comprehensive PyTest suite covering text parsing, chunk segmentation, and evaluation calculations:
```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```
*Result: 16/16 Unit Tests Passed.*

### Integration Smoke Test
Execute the end-to-end API pipeline test:
```bash
cd backend
source venv/bin/activate
python tests/integration_test.py
```
*Result: 27/27 Integration Checks Passed.*

### Production UI Build
Validate TypeScript compiler types and production bundle assets:
```bash
cd frontend
npm run build
```
*Result: Clean compilation (`dist/` generated with zero errors).*

---

## 7. Included Sample Dataset

Pre-packaged in `backend/data/sample_docs/`:
- `rag_survey_excerpt.txt`: Adapted from Gao et al., *"Retrieval-Augmented Generation for Large Language Models: A Survey"*, arXiv:2312.10997 (CC BY 4.0).
- `hallucination_survey_excerpt.txt`: Adapted from Zhang et al., *"Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models"*, arXiv:2309.01219 (CC BY 4.0).

---

## 8. Reproducibility & Research Integrity

- **Deterministic Seeds:** FAISS cosine similarity searches are deterministic.
- **Model Traceability:** Every Q&A session records `model_used`, `retrieval_mode`, `latency_ms`, and chunk identifiers.
- **No Synthesized Data:** All analytics and visualizations strictly query live experiment results stored in `sqlite:///./data/research.db`.
- **Honest Abstention:** Prompts instruct RAG models to state *"The provided evidence is insufficient to answer this question"* when context similarity falls below threshold.

---

## 9. References

- **Gao, Y. et al. (2023).** *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.
- **Zhang, Y. et al. (2023).** *Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.* arXiv:2309.01219.
- **Es, S. et al. (2023).** *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2305.14283.
- **Lewis, P. et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- **Xiao, S. et al. (2023).** *C-Pack: Packaged Resources to Advance General Chinese Embedding (BGE Embeddings).* arXiv:2309.07597.

---

## 10. License

Released under the [MIT License](LICENSE).
