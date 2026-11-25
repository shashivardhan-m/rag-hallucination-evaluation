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

        # 2. Entity & Number Fabulation
        cap_entities, numbers = cls.extract_entities_and_numbers(claim)
        for num in numbers:
            if num not in context:
                return "entity_fabulation"
        for ent in cap_entities:
            if len(ent) > 3 and ent.lower() not in context_clean:
                return "entity_fabulation"

        # 3. Intrinsic Contradiction (lexical negation or opposition cues)
        contradiction_cues = [
            ("increase", "decrease"), ("increase", "slow"), ("increase", "slowdown"),
            ("higher", "lower"), ("slow", "fast"), ("slowdown", "acceleration"),
            ("reversal", "continuation"), ("always", "never"), ("prevent", "cause"),
            ("inhibit", "activate"), ("not", "is"), ("none", "all"), ("halt", "accelerate"),
            ("accelerat", "slow"), ("up", "down"), ("reversed", "maintained"),
        ]
        for term_a, term_b in contradiction_cues:
            if (term_a in claim_clean and term_b in context_clean) or (term_b in claim_clean and term_a in context_clean):
                return "intrinsic_contradiction"

        # 4. Cross-Chunk Confusion (if multiple chunks present with overlapping topics)
        if " and " in claim_clean and len(context.split("\n\n")) > 1:
            chunks = context.split("\n\n")
            words = [w for w in claim_clean.split() if len(w) > 4]
            # If parts of the claim hit different chunks while conflicting
            chunk_hits = [sum(1 for w in words if w in c.lower()) for c in chunks]
            if sum(1 for h in chunk_hits if h > 0) >= 2:
                return "cross_chunk_confusion"

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
        Verify claims individually, calculate faithfulness score, and categorize any unentailed claims
        into the formal hallucination error taxonomy.
        """
        if not answer.strip():
            return 0.0, [], {k: 0.0 for k in HALLUCINATION_TYPES}

        if not context_chunks:
            return 0.0, [], {k: 0.0 for k in HALLUCINATION_TYPES}

        lower_ans = answer.lower()
        if (
            "cannot find sufficient information" in lower_ans
            or "not mentioned in the provided" in lower_ans
            or "insufficient information in the provided documents" in lower_ans
        ):
            # Explicit honest abstention
            return 1.0, [{"claim": answer, "status": "entailed", "error_type": None}], {k: 0.0 for k in HALLUCINATION_TYPES}

        claims = split_into_sentences(answer)
        if not claims:
            return 1.0, [], {k: 0.0 for k in HALLUCINATION_TYPES}

        nli_model = self._get_nli_model()
        full_context = " ".join([c.strip() for c in context_chunks if c.strip()])

        entailed_count = 0
        claims_analysis: List[Dict[str, Any]] = []
        error_counts = {k: 0 for k in HALLUCINATION_TYPES}

        for claim in claims:
            if len(claim.split()) < 3:
                # Trivial filler sentence
                entailed_count += 1
                claims_analysis.append({"claim": claim, "status": "entailed", "error_type": None})
                continue

            is_entailed = False
            if nli_model is not None:
                try:
                    scores = nli_model.predict([(full_context, claim)])[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    # Label 1 is typically entailment
                    if probs[1] >= 0.45 or int(np.argmax(probs)) == 1:
                        is_entailed = True
                except Exception:
                    pass

            if not is_entailed:
                # Heuristic alignment check
                claim_words = [w for w in clean_text(claim).lower().split() if len(w) > 3]
                context_words = set(clean_text(full_context).lower().split())
                if claim_words:
                    match_ratio = sum(1 for w in claim_words if w in context_words) / len(claim_words)
                    if match_ratio >= 0.70:
                        is_entailed = True

            if is_entailed:
                entailed_count += 1
                claims_analysis.append({"claim": claim, "status": "entailed", "error_type": None})
            else:
                err_type = HallucinationClassifier.classify_unsupported_claim(
                    claim=claim,
                    context=full_context,
                    context_precision=context_precision,
                )
                error_counts[err_type] += 1
                claims_analysis.append({"claim": claim, "status": "hallucinated", "error_type": err_type})

        faithfulness = clamp_score(safe_divide(entailed_count, len(claims), default=0.0))
        total_hallucinated = len(claims) - entailed_count
        taxonomy_proportions = {}
        for k, count in error_counts.items():
            prop = safe_divide(count, total_hallucinated, default=0.0) if total_hallucinated > 0 else 0.0
            taxonomy_proportions[k] = round(prop, 4)

        return faithfulness, claims_analysis, taxonomy_proportions

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

    # ------------------------------------------------------------------
    # Reference-Based Metrics
    # ------------------------------------------------------------------

    def compute_answer_correctness(
        self,
        answer: str,
        reference: str,
    ) -> Dict[str, float]:
        """Compute ROUGE-L, BERTScore F1, and a combined score against reference answer."""
        if not answer.strip() or not reference.strip():
            return {
                "rouge_l_f1": 0.0,
                "bertscore_f1": 0.0,
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

        rouge_l_f1 = clamp_score(rouge_l_f1)
        bertscore_f1 = clamp_score(bertscore_f1)
        combined = 0.4 * rouge_l_f1 + 0.6 * bertscore_f1

        return {
            "rouge_l_f1": round(rouge_l_f1, 4),
            "bertscore_f1": round(bertscore_f1, 4),
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
        faithfulness, claims_analysis, error_taxonomy = self.evaluate_claims(
            answer=answer,
            context_chunks=context_chunks,
            context_precision=ctx_precision,
        )
        hallucination_rate = self.compute_hallucination_rate(faithfulness)
        ans_relevance = self.compute_answer_relevance(question, answer)
        ret_precision = self.compute_retrieval_precision(context_chunks, question)

        metrics: Dict[str, float] = {
            "faithfulness": round(faithfulness, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "answer_relevance": round(ans_relevance, 4),
            "context_precision": round(ctx_precision, 4),
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
            "claims_count": len(split_into_sentences(answer)),
            "claims_analysis": claims_analysis,
            "error_taxonomy": error_taxonomy,
        }

        return {
            "metrics": metrics,
            "error_taxonomy": error_taxonomy,
            "details": details,
        }


def get_evaluator() -> HallucinationEvaluator:
    """Return singleton HallucinationEvaluator instance."""
    return HallucinationEvaluator()
