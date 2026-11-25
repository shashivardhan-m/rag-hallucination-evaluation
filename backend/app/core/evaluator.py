"""Comprehensive RAG hallucination and retrieval quality evaluation engine with fine-grained error taxonomy."""

import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None  # type: ignore
    _CROSS_ENCODER_AVAILABLE = False

from app.core.embeddings import EmbeddingManager, get_embedding_manager
from app.utils.metrics_utils import (
    clamp_score,
    compute_mrr,
    compute_ndcg,
    compute_rank_sensitive_context_precision,
    compute_rouge_l_native,
    safe_divide,
)
from app.utils.text_utils import clean_text, split_into_sentences


# Standard Hallucination Taxonomy
HALLUCINATION_TYPES = {
    "intrinsic_contradiction": "Claim directly contradicts facts asserted in the retrieved context",
    "extrinsic_extrapolation": "Claim introduces speculative or ungrounded assertions not in context",
    "entity_fabulation": "Fabrication of non-existent entities, dates, or numerical metrics",
    "cross_chunk_confusion": "Erroneous synthesis or attribute confusion across disparate retrieved chunks",
    "refusal_failure": "Failure to abstain/refuse when the retrieved context is irrelevant or uninformative",
}


class HallucinationClassifier:
    """Classifies unsupported claims into the formal 5-category hallucination taxonomy."""

    @staticmethod
    def extract_entities_and_numbers(text: str) -> Tuple[List[str], List[str]]:
        """Extract potential named entities (capitalized phrases) and numerical tokens."""
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        words = text.split()
        capitalized = [
            w.strip(".,;:?!'\"()[]{}")
            for w in words
            if len(w) > 1 and w[0].isupper() and not w.isupper()
        ]
        return capitalized, numbers

    @classmethod
    def classify_unsupported_claim(
        cls,
        claim: str,
        context: str,
        context_precision: float = 0.5,
    ) -> str:
        """Categorize an unsupported claim into the 5 academic error categories."""
        claim_clean = claim.lower()
        context_clean = context.lower()

        # 1. Refusal Failure / Failure to Abstain
        if context_precision < 0.20 or not context.strip():
            return "refusal_failure"

        # 2. Intrinsic Contradiction (lexical negation, antonyms, or conflicting numerical attributes)
        contradiction_cues = [
            ("increase", "decrease"), ("increase", "slow"), ("increase", "slowdown"),
            ("higher", "lower"), ("slow", "fast"), ("slowdown", "acceleration"),
            ("reversal", "continuation"), ("always", "never"), ("prevent", "cause"),
            ("inhibit", "activate"), ("not", "is"), ("none", "all"), ("halt", "accelerate"),
            ("accelerat", "slow"), ("up", "down"), ("reversed", "maintained"),
            ("orbits", "landed"), ("never", "was"),
        ]
        for term_a, term_b in contradiction_cues:
            if (term_a in claim_clean and term_b in context_clean) or (term_b in claim_clean and term_a in context_clean):
                return "intrinsic_contradiction"

        # Check for conflicting numerical values on shared predicates (e.g., launched in 2021 vs 1990)
        cap_entities, numbers = cls.extract_entities_and_numbers(claim)
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context))
        shared_predicates = [
            w for w in claim_clean.split()
            if len(w) > 4 and w in context_clean
        ]
        if numbers and context_numbers and len(shared_predicates) >= 2 and not any(n in context_numbers for n in numbers):
            return "intrinsic_contradiction"

        # 3. Cross-Chunk Confusion (if multiple chunks present with overlapping topics)
        if " and " in claim_clean and len(context.split("\n\n")) > 1:
            chunks = context.split("\n\n")
            words = [w for w in claim_clean.split() if len(w) > 4]
            # If parts of the claim hit different chunks while conflicting
            chunk_hits = [sum(1 for w in words if w in c.lower()) for c in chunks]
            if sum(1 for h in chunk_hits if h > 0) >= 2:
                return "cross_chunk_confusion"

        # 4. Entity & Number Fabulation (novel ungrounded entities/numbers without explicit conflict)
        for num in numbers:
            if num not in context:
                return "entity_fabulation"
        for ent in cap_entities:
            if len(ent) > 3 and ent.lower() not in context_clean:
                return "entity_fabulation"

        # 5. Extrinsic Extrapolation (default parametric leak)
        return "extrinsic_extrapolation"


class HallucinationEvaluator:
    """Calculates reference-free and reference-based evaluation metrics and categorizes errors."""

    _instance: Optional["HallucinationEvaluator"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "HallucinationEvaluator":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(HallucinationEvaluator, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.device = "cpu"
        self.embedding_manager = get_embedding_manager()
        self.nli_model_name = "cross-encoder/nli-deberta-v3-small"
        self._nli_model: Any = None
        self._nli_lock = threading.Lock()
        self._rouge_scorer = None
        self._initialized = True

    def _get_nli_model(self) -> Any:
        """Lazy load NLI CrossEncoder model for claim entailment verification."""
        with self._nli_lock:
            if self._nli_model is None and _CROSS_ENCODER_AVAILABLE and CrossEncoder is not None:
                try:
                    self._nli_model = CrossEncoder(self.nli_model_name, device=self.device)
                except Exception:
                    self._nli_model = None
            return self._nli_model

    def _get_rouge_scorer(self):
        """Lazy load ROUGE scorer."""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            except Exception:
                self._rouge_scorer = None
        return self._rouge_scorer

    # ------------------------------------------------------------------
    # Reference-Free Metrics & Claim-Level Grounding
    # ------------------------------------------------------------------

    def evaluate_claims(
        self,
        answer: str,
        context_chunks: List[str],
        context_precision: float = 0.5,
    ) -> Tuple[float, List[Dict[str, Any]], Dict[str, float]]:
        """
        Verify claims individually using 3-way NLI classification (Entailment, Contradiction, Neutral),
        calculate faithfulness score, and categorize unentailed claims into the formal error taxonomy.
        Safely handles empty answers or zero extracted claims without ZeroDivisionError.
        """
        breakdown = self.compute_claim_breakdown(
            answer=answer,
            context_chunks=context_chunks,
            context_precision=context_precision,
        )
        return (
            breakdown["faithfulness"],
            breakdown["claims_analysis"],
            breakdown["taxonomy_proportions"],
        )

    def compute_claim_breakdown(
        self,
        answer: str,
        context_chunks: List[str],
        context_precision: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Formal 3-way NLI Claim Verification:
        1. Supported claim rate: Proportion of claims entailed by context (Factual Grounding).
        2. Contradiction rate: Proportion of claims directly conflicting with context.
        3. Unsupported claim rate: Proportion of claims lacking sufficient grounding (Extrinsic Neutral).
        4. Faithfulness: Overall grounding of claims (= supported claim rate).

        Safely handles empty claims and answers with zero extracted claims (preventing division-by-zero).
        """
        empty_taxonomy = {k: 0.0 for k in HALLUCINATION_TYPES}
        if not answer.strip():
            return {
                "faithfulness": 0.0,
                "supported_claim_rate": 0.0,
                "contradiction_rate": 0.0,
                "unsupported_claim_rate": 0.0,
                "total_claims": 0,
                "supported_claims_count": 0,
                "contradiction_claims_count": 0,
                "unsupported_claims_count": 0,
                "claims_analysis": [],
                "taxonomy_proportions": empty_taxonomy,
            }

        if not context_chunks:
            claim_count = len(split_into_sentences(answer))
            return {
                "faithfulness": 0.0,
                "supported_claim_rate": 0.0,
                "contradiction_rate": 0.0,
                "unsupported_claim_rate": 1.0,
                "total_claims": claim_count,
                "supported_claims_count": 0,
                "contradiction_claims_count": 0,
                "unsupported_claims_count": claim_count,
                "claims_analysis": [],
                "taxonomy_proportions": empty_taxonomy,
            }

        lower_ans = answer.lower()
        if (
            "cannot find sufficient information" in lower_ans
            or "not mentioned in the provided" in lower_ans
            or "insufficient information in the provided documents" in lower_ans
        ):
            # Explicit honest abstention
            return {
                "faithfulness": 1.0,
                "supported_claim_rate": 1.0,
                "contradiction_rate": 0.0,
                "unsupported_claim_rate": 0.0,
                "total_claims": 1,
                "supported_claims_count": 1,
                "contradiction_claims_count": 0,
                "unsupported_claims_count": 0,
                "claims_analysis": [{"claim": answer, "status": "entailed", "nli_label": "entailment", "error_type": None}],
                "taxonomy_proportions": empty_taxonomy,
            }

        claims = split_into_sentences(answer)
        if not claims:
            # Safe zero-claim handling: Non-assertive answer contains 0 unsupported claims
            return {
                "faithfulness": 1.0,
                "supported_claim_rate": 1.0,
                "contradiction_rate": 0.0,
                "unsupported_claim_rate": 0.0,
                "total_claims": 0,
                "supported_claims_count": 0,
                "contradiction_claims_count": 0,
                "unsupported_claims_count": 0,
                "claims_analysis": [],
                "taxonomy_proportions": empty_taxonomy,
            }

        nli_model = self._get_nli_model()
        full_context = " ".join([c.strip() for c in context_chunks if c.strip()])

        entailed_count = 0
        contradiction_count = 0
        unsupported_count = 0

        claims_analysis: List[Dict[str, Any]] = []
        error_counts = {k: 0 for k in HALLUCINATION_TYPES}

        for claim in claims:
            if len(claim.split()) < 3:
                # Trivial filler sentence treated as non-assertive grounded
                entailed_count += 1
                claims_analysis.append({
                    "claim": claim,
                    "status": "entailed",
                    "nli_label": "entailment",
                    "error_type": None,
                })
                continue

            nli_label = "neutral"
            if nli_model is not None:
                try:
                    scores = nli_model.predict([(full_context, claim)])[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    pred_idx = int(np.argmax(probs))
                    # MNLI standard mapping: 0=contradiction, 1=entailment, 2=neutral
                    if probs[1] >= 0.45 or pred_idx == 1:
                        nli_label = "entailment"
                    elif probs[0] >= 0.45 or pred_idx == 0:
                        nli_label = "contradiction"
                    else:
                        nli_label = "neutral"
                except Exception:
                    nli_label = "neutral"

            if nli_label == "neutral":
                # Deterministic lexical, numerical, and semantic heuristic fallback
                claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim))
                context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", full_context))
                if claim_numbers and not claim_numbers.issubset(context_numbers):
                    # Direct numerical conflict / distortion
                    nli_label = "contradiction"

                if nli_label == "neutral":
                    claim_err = HallucinationClassifier.classify_unsupported_claim(
                        claim=claim,
                        context=full_context,
                        context_precision=context_precision,
                    )
                    if claim_err == "intrinsic_contradiction":
                        nli_label = "contradiction"

                if nli_label == "neutral":
                    clean_tokens = re.findall(r"\b[a-zA-Z]{3,}\b", claim.lower())
                    ctx_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", full_context.lower()))
                    ctx_roots = {w[:4] for w in ctx_tokens if len(w) >= 4} | ctx_tokens
                    if clean_tokens:
                        matched = sum(1 for w in clean_tokens if w in ctx_tokens or (len(w) >= 4 and w[:4] in ctx_roots))
                        match_ratio = matched / len(clean_tokens)
                        if match_ratio >= 0.50:
                            nli_label = "entailment"

            if nli_label == "entailment":
                entailed_count += 1
                claims_analysis.append({
                    "claim": claim,
                    "status": "entailed",
                    "nli_label": "entailment",
                    "error_type": None,
                })
            elif nli_label == "contradiction":
                contradiction_count += 1
                error_counts["intrinsic_contradiction"] += 1
                claims_analysis.append({
                    "claim": claim,
                    "status": "contradiction",
                    "nli_label": "contradiction",
                    "error_type": "intrinsic_contradiction",
                })
            else:  # neutral / unsupported
                unsupported_count += 1
                err_type = HallucinationClassifier.classify_unsupported_claim(
                    claim=claim,
                    context=full_context,
                    context_precision=context_precision,
                )
                error_counts[err_type] += 1
                claims_analysis.append({
                    "claim": claim,
                    "status": "unsupported",
                    "nli_label": "neutral",
                    "error_type": err_type,
                })

        total_claims = len(claims)
        # Mathematical guarantees against zero division:
        supported_claim_rate = clamp_score(safe_divide(entailed_count, total_claims, default=1.0 if total_claims == 0 else 0.0))
        contradiction_rate = clamp_score(safe_divide(contradiction_count, total_claims, default=0.0))
        unsupported_claim_rate = clamp_score(safe_divide(unsupported_count, total_claims, default=0.0))
        faithfulness = supported_claim_rate

        total_hallucinated = contradiction_count + unsupported_count
        taxonomy_proportions = {}
        for k, count in error_counts.items():
            prop = safe_divide(count, total_hallucinated, default=0.0) if total_hallucinated > 0 else 0.0
            taxonomy_proportions[k] = round(prop, 4)

        return {
            "faithfulness": round(faithfulness, 4),
            "supported_claim_rate": round(supported_claim_rate, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "total_claims": total_claims,
            "supported_claims_count": entailed_count,
            "contradiction_claims_count": contradiction_count,
            "unsupported_claims_count": unsupported_count,
            "claims_analysis": claims_analysis,
            "taxonomy_proportions": taxonomy_proportions,
        }

    def compute_faithfulness(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        """NLI-based Faithfulness score [0.0 - 1.0]."""
        faithfulness, _, _ = self.evaluate_claims(answer, context_chunks)
        return faithfulness

    def compute_answer_relevance(
        self,
        question: str,
        answer: str,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> float:
        """Cosine similarity between question and generated answer embeddings."""
        if not question.strip() or not answer.strip():
            return 0.0

        em = embedding_manager or self.embedding_manager
        q_emb = em.get_query_embedding(question)
        a_emb = em.get_query_embedding(answer)

        cosine_sim = float(np.dot(q_emb, a_emb))
        return clamp_score(cosine_sim)

    def compute_context_precision(
        self,
        question: str,
        chunks: List[str],
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> float:
        """Average cosine similarity of retrieved context chunks to the user's question."""
        if not question.strip() or not chunks:
            return 0.0

        em = embedding_manager or self.embedding_manager
        q_emb = em.get_query_embedding(question)
        chunk_embs = em.get_embeddings(chunks)

        if chunk_embs.size == 0:
            return 0.0

        similarities = np.dot(chunk_embs, q_emb)
        avg_sim = float(np.mean(similarities))
        return clamp_score(avg_sim)

    def compute_retrieval_precision(
        self,
        chunks: List[str],
        question: str,
        threshold: float = 0.4,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> float:
        """Fraction of retrieved chunks exceeding semantic relevance threshold."""
        if not question.strip() or not chunks:
            return 0.0

        em = embedding_manager or self.embedding_manager
        q_emb = em.get_query_embedding(question)
        chunk_embs = em.get_embeddings(chunks)

        if chunk_embs.size == 0:
            return 0.0

        similarities = np.dot(chunk_embs, q_emb)
        relevant_count = sum(1 for sim in similarities if sim >= threshold)
        return clamp_score(safe_divide(relevant_count, len(chunks)))

    def compute_rank_sensitive_context_precision(
        self,
        chunks: List[str],
        question: str,
        retrieved_ids: Optional[List[str]] = None,
        relevant_ids: Optional[List[str]] = None,
        reference: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculates formal rank-sensitive Information Retrieval metrics:
        1. Context Precision@K: (1 / min(K, total_relevant)) * sum_{i=1}^K (Precision@i * v_i)
        2. Mean Reciprocal Rank (MRR@K): 1 / rank_first_relevant
        3. Normalized Discounted Cumulative Gain (nDCG@K): DCG@K / IDCG@K

        Returns (context_precision_at_k, mrr_at_k, ndcg_at_k).
        """
        if not chunks:
            return 0.0, 0.0, 0.0

        relevance_vector: List[int] = []
        if retrieved_ids and relevant_ids:
            relevant_set = set(relevant_ids)
            relevance_vector = [1 if cid in relevant_set else 0 for cid in retrieved_ids]
        else:
            # Evaluate relevance using semantic similarity threshold to question/reference
            target_text = reference if reference and reference.strip() else question
            if not target_text.strip():
                return 0.0, 0.0, 0.0
            q_emb = self.embedding_manager.get_query_embedding(target_text)
            chunk_embs = self.embedding_manager.get_embeddings(chunks)
            if chunk_embs.size == 0:
                return 0.0, 0.0, 0.0
            sims = np.dot(chunk_embs, q_emb)
            # Binary relevance at threshold 0.35
            relevance_vector = [1 if float(s) >= 0.35 else 0 for s in sims]

        total_rel = len(relevant_ids) if relevant_ids else max(1, sum(relevance_vector))
        k_val = k or len(chunks)

        ctx_prec = compute_rank_sensitive_context_precision(
            retrieved_chunk_relevance=relevance_vector,
            total_relevant=total_rel,
            k=k_val,
        )
        mrr = compute_mrr(retrieved_chunk_relevance=relevance_vector, k=k_val)
        ndcg = compute_ndcg(retrieved_chunk_relevance=relevance_vector, k=k_val)

        return round(ctx_prec, 4), round(mrr, 4), round(ndcg, 4)

    # ------------------------------------------------------------------
    # Reference-Based Metrics
    # ------------------------------------------------------------------

    def compute_factual_correctness(
        self,
        answer: str,
        reference: str,
    ) -> float:
        """
        NLI-based factual claim verification against ground-truth reference answer.
        Unlike surface text similarity (ROUGE) or soft embeddings (BERTScore),
        factual correctness verifies whether each discrete factual assertion is strictly
        supported by the reference, with strict penalties for numerical and entity mutations.
        """
        if not answer.strip() or not reference.strip():
            return 0.0

        claims = split_into_sentences(answer)
        if not claims:
            return 1.0

        nli_model = self._get_nli_model()
        verified_claims = 0

        # Extract numerical tokens for entity-level verification
        ref_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", reference))

        for claim in claims:
            if len(claim.split()) < 3:
                verified_claims += 1
                continue

            # Numerical conflict check: mutated numbers invalidate claim correctness
            claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim))
            if claim_numbers and not claim_numbers.issubset(ref_numbers):
                continue

            is_entailed = False
            if nli_model is not None:
                try:
                    scores = nli_model.predict([(reference, claim)])[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    if probs[1] >= 0.45 or int(np.argmax(probs)) == 1:
                        is_entailed = True
                except Exception:
                    pass

            if not is_entailed:
                # Heuristic word overlap against reference
                claim_words = [w for w in clean_text(claim).lower().split() if len(w) > 3]
                ref_words = set(clean_text(reference).lower().split())
                if claim_words:
                    overlap = sum(1 for w in claim_words if w in ref_words) / len(claim_words)
                    if overlap >= 0.65:
                        is_entailed = True

            if is_entailed:
                verified_claims += 1

        return clamp_score(safe_divide(verified_claims, len(claims), default=0.0))

    def compute_answer_correctness(
        self,
        answer: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Multi-component Answer Correctness Evaluation:
        1. ROUGE-L: Lexical overlap via longest common subsequence (LCS n-gram recall/precision).
        2. BERTScore: Dense semantic similarity in embedding space.
        3. Factual Correctness: NLI-grounded claim-level entailment and numerical validation.

        Theoretical Justification for Weights (0.20 ROUGE-L, 0.30 BERTScore, 0.50 Factual Correctness):
        Surface lexical overlap and soft embeddings are susceptible to false positives (e.g. inverted
        dates, swapped negation, or mutated quantities retain high BERTScore). Factual claim entailment
        serves as the primary anchor (50%), supplemented by semantic coherence (30%) and lexical overlap (20%).
        """
        if not answer.strip() or not reference.strip():
            return {
                "rouge_l_f1": 0.0,
                "bertscore_f1": 0.0,
                "factual_correctness": 0.0,
                "answer_correctness": 0.0,
            }

        # 1. ROUGE-L (with native pure-Python LCS fallback)
        rouge_l_f1 = 0.0
        try:
            scorer = self._get_rouge_scorer()
            if scorer:
                scores = scorer.score(reference, answer)
                rouge_l_f1 = float(scores["rougeL"].fmeasure)
            else:
                rouge_l_f1 = compute_rouge_l_native(reference, answer)
        except Exception:
            rouge_l_f1 = compute_rouge_l_native(reference, answer)

        # 2. Semantic Embedding Similarity (BERTScore surrogate)
        bertscore_f1 = 0.0
        try:
            import bert_score
            p, r, f1 = bert_score.score(
                [answer],
                [reference],
                lang="en",
                verbose=False,
                rescale_with_baseline=False,
            )
            bertscore_f1 = float(f1[0])
        except Exception:
            try:
                emb_ans = self.embedding_manager.get_query_embedding(answer)
                emb_ref = self.embedding_manager.get_query_embedding(reference)
                bertscore_f1 = float(np.dot(emb_ans, emb_ref))
            except Exception:
                bertscore_f1 = rouge_l_f1

        # 3. Factual Claim Verification (NLI + Entity consistency)
        factual_correctness = self.compute_factual_correctness(answer, reference)

        rouge_l_f1 = clamp_score(rouge_l_f1)
        bertscore_f1 = clamp_score(bertscore_f1)
        factual_correctness = clamp_score(factual_correctness)

        combined = 0.20 * rouge_l_f1 + 0.30 * bertscore_f1 + 0.50 * factual_correctness

        return {
            "rouge_l_f1": round(rouge_l_f1, 4),
            "bertscore_f1": round(bertscore_f1, 4),
            "factual_correctness": round(factual_correctness, 4),
            "answer_correctness": round(combined, 4),
        }

    def compute_context_recall(
        self,
        question: str,
        chunks: List[str],
        reference: str,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> float:
        """Measure coverage of reference answer content by the concatenated retrieved context."""
        if not reference.strip() or not chunks:
            return 0.0

        em = embedding_manager or self.embedding_manager
        concat_context = " ".join([c.strip() for c in chunks if c.strip()])

        ref_emb = em.get_query_embedding(reference)
        context_emb = em.get_query_embedding(concat_context[:2048])

        score = float(np.dot(ref_emb, context_emb))
        return clamp_score(score)

    def compute_retrieval_recall(
        self,
        retrieved_ids: Optional[List[str]],
        relevant_ids: Optional[List[str]],
    ) -> float:
        """Standard Recall@K: fraction of ground-truth relevant document IDs successfully retrieved."""
        if not relevant_ids:
            return 0.0

        if not retrieved_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        intersect = retrieved_set.intersection(relevant_set)
        return clamp_score(safe_divide(len(intersect), len(relevant_set)))

    @staticmethod
    def compute_hallucination_rate(faithfulness: float) -> float:
        """Hallucination Rate is 1.0 - faithfulness."""
        f = clamp_score(faithfulness)
        return round(1.0 - f, 4)

    def evaluate_qa_pair(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        reference: Optional[str] = None,
        retrieved_ids: Optional[List[str]] = None,
        relevant_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute all applicable reference-free, reference-based, and error taxonomy metrics."""
        ctx_precision = self.compute_context_precision(question, context_chunks)
        rank_ctx_prec, mrr_score, ndcg_score = self.compute_rank_sensitive_context_precision(
            chunks=context_chunks,
            question=question,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            reference=reference,
        )

        claim_breakdown = self.compute_claim_breakdown(
            answer=answer,
            context_chunks=context_chunks,
            context_precision=ctx_precision,
        )
        faithfulness = claim_breakdown["faithfulness"]
        hallucination_rate = self.compute_hallucination_rate(faithfulness)
        ans_relevance = self.compute_answer_relevance(question, answer)
        ret_precision = self.compute_retrieval_precision(context_chunks, question)

        metrics: Dict[str, float] = {
            "faithfulness": round(faithfulness, 4),
            "supported_claim_rate": round(claim_breakdown["supported_claim_rate"], 4),
            "contradiction_rate": round(claim_breakdown["contradiction_rate"], 4),
            "unsupported_claim_rate": round(claim_breakdown["unsupported_claim_rate"], 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "answer_relevance": round(ans_relevance, 4),
            "context_precision": round(ctx_precision, 4),
            "rank_sensitive_context_precision": round(rank_ctx_prec, 4),
            "mrr": round(mrr_score, 4),
            "ndcg": round(ndcg_score, 4),
            "retrieval_precision": round(ret_precision, 4),
        }

        # Reference-based metrics (if reference provided)
        if reference and reference.strip():
            correctness_dict = self.compute_answer_correctness(answer, reference)
            metrics.update(correctness_dict)

            ctx_recall = self.compute_context_recall(question, context_chunks, reference)
            metrics["context_recall"] = round(ctx_recall, 4)

        if relevant_ids:
            ret_recall = self.compute_retrieval_recall(retrieved_ids, relevant_ids)
            metrics["retrieval_recall"] = round(ret_recall, 4)

        details = {
            "num_context_chunks": len(context_chunks),
            "has_reference": bool(reference and reference.strip()),
            "claims_count": claim_breakdown.get("total_claims", 0),
            "supported_claims_count": claim_breakdown.get("supported_claims_count", 0),
            "contradiction_claims_count": claim_breakdown.get("contradiction_claims_count", 0),
            "unsupported_claims_count": claim_breakdown.get("unsupported_claims_count", 0),
            "claims_analysis": claim_breakdown.get("claims_analysis", []),
            "error_taxonomy": claim_breakdown.get("taxonomy_proportions", {}),
        }

        return {
            "metrics": metrics,
            "error_taxonomy": claim_breakdown["taxonomy_proportions"],
            "details": details,
        }


def get_evaluator() -> HallucinationEvaluator:
    """Return singleton HallucinationEvaluator instance."""
    return HallucinationEvaluator()
