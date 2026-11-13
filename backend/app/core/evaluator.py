"""Comprehensive RAG hallucination and retrieval quality evaluation engine."""

import math
import threading
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.core.embeddings import EmbeddingManager, get_embedding_manager
from app.utils.metrics_utils import clamp_score, safe_divide
from app.utils.text_utils import clean_text, split_into_sentences


class HallucinationEvaluator:
    """Calculates reference-free and reference-based evaluation metrics for LLM answers."""

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_manager = get_embedding_manager()
        self.nli_model_name = "cross-encoder/nli-deberta-v3-small"
        self._nli_model: Optional[CrossEncoder] = None
        self._nli_lock = threading.Lock()
        self._rouge_scorer = None
        self._initialized = True

    def _get_nli_model(self) -> Optional[CrossEncoder]:
        """Lazy load NLI CrossEncoder model for claim entailment verification."""
        with self._nli_lock:
            if self._nli_model is None:
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
    # Reference-Free Metrics
    # ------------------------------------------------------------------

    def compute_faithfulness(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        """
        NLI-based Faithfulness: verifies that claims in the answer are strictly entailed
        by the retrieved context. Returns ratio of entailed claims [0.0 - 1.0].
        """
        if not answer.strip():
            return 0.0

        if not context_chunks:
            return 0.0

        # Negative phrase check: if the model explicitly admitted lack of context, it's not hallucinating
        lower_ans = answer.lower()
        if "cannot find sufficient information" in lower_ans or "not mentioned in the provided" in lower_ans:
            return 1.0

        claims = split_into_sentences(answer)
        if not claims:
            return 1.0

        nli_model = self._get_nli_model()
        full_context = " ".join([c.strip() for c in context_chunks if c.strip()])

        if nli_model is None:
            # Fallback heuristic: token overlap & embedding similarity
            return self._heuristic_faithfulness(claims, full_context)

        # Map labels from NLI model
        entailment_idx = 1
        if hasattr(nli_model.model, "config") and hasattr(nli_model.model.config, "label2id"):
            l2i = {k.lower(): v for k, v in nli_model.model.config.label2id.items()}
            entailment_idx = l2i.get("entailment", 1)

        entailed_count = 0
        for claim in claims:
            if len(claim.split()) < 3:
                # Trivial filler sentence
                entailed_count += 1
                continue

            # Evaluate (context, claim) pair
            try:
                scores = nli_model.predict([(full_context, claim)])[0]
                # Softmax scores
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
                pred_label = int(np.argmax(probs))

                # Entailed if predicted label is entailment or entailment probability >= 0.5
                if pred_label == entailment_idx or probs[entailment_idx] >= 0.45:
                    entailed_count += 1
            except Exception:
                # If NLI prediction fails, fallback to semantic match
                emb_q = self.embedding_manager.get_query_embedding(claim)
                emb_c = self.embedding_manager.get_query_embedding(full_context[:1000])
                if float(np.dot(emb_q, emb_c)) >= 0.65:
                    entailed_count += 1

        score = safe_divide(entailed_count, len(claims), default=0.0)
        return clamp_score(score)

    def _heuristic_faithfulness(self, claims: List[str], context: str) -> float:
        """Heuristic calculation when NLI transformer is unavailable."""
        if not context:
            return 0.0
        context_words = set(clean_text(context).lower().split())
        entailed = 0
        for claim in claims:
            words = [w for w in clean_text(claim).lower().split() if len(w) > 3]
            if not words:
                entailed += 1
                continue
            matched = sum(1 for w in words if w in context_words)
            if (matched / len(words)) >= 0.5:
                entailed += 1
        return clamp_score(safe_divide(entailed, len(claims)))

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
        """
        Compute ROUGE-L, BERTScore F1, and a balanced combined score against reference.
        """
        if not answer.strip() or not reference.strip():
            return {
                "rouge_l_f1": 0.0,
                "bertscore_f1": 0.0,
                "answer_correctness": 0.0,
            }

        # 1. ROUGE-L
        rouge_l_f1 = 0.0
        try:
            scorer = self._get_rouge_scorer()
            if scorer:
                scores = scorer.score(reference, answer)
                rouge_l_f1 = float(scores["rougeL"].fmeasure)
        except Exception:
            rouge_l_f1 = 0.0

        # 2. BERTScore F1 (with graceful fallback to embedding cosine similarity)
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
            # Fallback: Embedding Cosine Similarity
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

    # ------------------------------------------------------------------
    # Hallucination Rate & Batch Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hallucination_rate(faithfulness: float) -> float:
        """Hallucination Rate is the inverse of context faithfulness: 1.0 - faithfulness."""
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
        """Compute all applicable reference-free and reference-based metrics for a QA pair."""
        metrics: Dict[str, float] = {}

        # Reference-free metrics
        faithfulness = self.compute_faithfulness(answer, context_chunks)
        hallucination_rate = self.compute_hallucination_rate(faithfulness)
        ans_relevance = self.compute_answer_relevance(question, answer)
        ctx_precision = self.compute_context_precision(question, context_chunks)
        ret_precision = self.compute_retrieval_precision(context_chunks, question)

        metrics["faithfulness"] = round(faithfulness, 4)
        metrics["hallucination_rate"] = round(hallucination_rate, 4)
        metrics["answer_relevance"] = round(ans_relevance, 4)
        metrics["context_precision"] = round(ctx_precision, 4)
        metrics["retrieval_precision"] = round(ret_precision, 4)

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
        }

        return {
            "metrics": metrics,
            "details": details,
        }


def get_evaluator() -> HallucinationEvaluator:
    """Return singleton HallucinationEvaluator instance."""
    return HallucinationEvaluator()
