# Formal Problem Statement: Hallucinations in Retrieval-Augmented Generation

## 1. Background and Motivation
Large Language Models (LLMs) trained on autoregressive parametric objectives frequently generate untruthful, ungrounded, or contradictory content—a failure mode termed **hallucination**. While Retrieval-Augmented Generation (RAG) was introduced to ground model outputs in external, verifiable reference passages, empirical evidence demonstrates that RAG systems do not inherently eliminate hallucinations. 

RAG pipelines introduce novel error surfaces:
1. **Retrieval Misses and Rank Depletion**: When relevant evidence is ranked poorly or omitted entirely, generator models default to parametric priors.
2. **Parametric-Contextual Contradictions**: When retrieved passages conflict with strong parametric pre-training memories, models often synthesize corrupted or contradictory outputs.
3. **Cross-Chunk Synthesizing Errors**: Information distributed across disjoint text chunks often suffers from coreference loss, leading to entity and attribute swaps.
4. **Distractor Sensitivity and Refusal Failures**: Generators frequently fail to abstain when retrieved passages are irrelevant or uninformative, manufacturing plausible-sounding extrapolations.

## 2. Research Objective
This research investigates the direct quantitative causal relationships between:
- Retrieval configurations (Dense inner-product search, BM25 lexical search, Hybrid Reciprocal Rank Fusion, and Cross-Encoder reranking),
- Context chunking granularities (128, 256, 512, and 1024 token windows), and
- Downstream factual consistency, fine-grained hallucination rates, and answer correctness across standardized benchmarks (**RAGTruth** and **HaluEval**).

## 3. Mathematical Formulation
Let $q \in \mathcal{Q}$ denote a query, and $\mathcal{D} = \{c_1, c_2, \dots, c_M\}$ denote the document corpus partitioned into chunks.
A retrieval pipeline $\mathcal{R}$ selects a top-$K$ subset $\mathcal{C}_K \subset \mathcal{D}$ ordered by relevance scores.
The generator model $\mathcal{G}_{\theta}$ produces an answer sequence:
$$a = \mathcal{G}_{\theta}(q, \mathcal{C}_K) = (w_1, w_2, \dots, w_T)$$

We decompose $a$ into a set of discrete, verifiable atomic claims $\mathcal{S}(a) = \{s_1, s_2, \dots, s_N\}$.
For each claim $s_i$, an Entailment model $\mathcal{E}$ evaluates its relation to the concatenated context $\mathcal{C}_K$:
$$\mathcal{E}(s_i, \mathcal{C}_K) \in \{\text{Entailment}, \text{Contradiction}, \text{Neutral}\}$$

The core metrics are formally defined:
- **Supported Claim Rate (Faithfulness $F$)**:
  $$F = \frac{|\{s_i \in \mathcal{S}(a) : \mathcal{E}(s_i, \mathcal{C}_K) = \text{Entailment}\}|}{\max(1, |\mathcal{S}(a)|)}$$
- **Contradiction Rate ($R_{\text{contra}}$)**:
  $$R_{\text{contra}} = \frac{|\{s_i \in \mathcal{S}(a) : \mathcal{E}(s_i, \mathcal{C}_K) = \text{Contradiction}\}|}{\max(1, |\mathcal{S}(a)|)}$$
- **Unsupported Claim Rate ($R_{\text{unsup}}$)**:
  $$R_{\text{unsup}} = \frac{|\{s_i \in \mathcal{S}(a) : \mathcal{E}(s_i, \mathcal{C}_K) = \text{Neutral}\}|}{\max(1, |\mathcal{S}(a)|)}$$
- **Hallucination Rate ($H$)**:
  $$H = 1.0 - F = R_{\text{contra}} + R_{\text{unsup}}$$
