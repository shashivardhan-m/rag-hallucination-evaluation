"""Unit tests for HallucinationEvaluator metrics using mock and synthetic test fixtures."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from app.core.evaluator import HallucinationEvaluator


@pytest.fixture
def evaluator() -> HallucinationEvaluator:
    return HallucinationEvaluator()


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager returning predictable normalized embeddings."""
    mock_em = MagicMock()

    def _mock_query_emb(text: str):
        # Generate deterministic vector based on string hash
        np.random.seed(abs(hash(text)) % (2**31))
        vec = np.random.randn(128).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def _mock_embs(texts: list[str]):
        return np.vstack([_mock_query_emb(t) for t in texts])

    mock_em.get_query_embedding.side_effect = _mock_query_emb
    mock_em.get_embeddings.side_effect = _mock_embs
    return mock_em


def test_compute_hallucination_rate(evaluator: HallucinationEvaluator):
    assert evaluator.compute_hallucination_rate(1.0) == 0.0
    assert evaluator.compute_hallucination_rate(0.0) == 1.0
    assert evaluator.compute_hallucination_rate(0.75) == 0.25
    assert evaluator.compute_hallucination_rate(0.8234) == round(1.0 - 0.8234, 4)


def test_compute_retrieval_recall(evaluator: HallucinationEvaluator):
    # Full recall
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = ["doc_1", "doc_2"]
    assert evaluator.compute_retrieval_recall(retrieved, relevant) == 1.0

    # Partial recall (1 out of 2)
    retrieved_partial = ["doc_1", "doc_4"]
    assert evaluator.compute_retrieval_recall(retrieved_partial, relevant) == 0.5

    # Zero recall
    retrieved_none = ["doc_8", "doc_9"]
    assert evaluator.compute_retrieval_recall(retrieved_none, relevant) == 0.0

    # Empty cases
    assert evaluator.compute_retrieval_recall([], relevant) == 0.0
    assert evaluator.compute_retrieval_recall(retrieved, []) == 0.0


def test_faithfulness_negative_rejection(evaluator: HallucinationEvaluator):
    # If the model explicitly stated it cannot find info, faithfulness is 1.0 (not hallucinated)
    answer = "I cannot find sufficient information in the provided documents to answer this question."
    score = evaluator.compute_faithfulness(answer, ["Context chunk about solar flares."])
    assert score == 1.0


def test_faithfulness_empty_inputs(evaluator: HallucinationEvaluator):
    assert evaluator.compute_faithfulness("", ["Some context"]) == 0.0
    assert evaluator.compute_faithfulness("Some answer claim.", []) == 0.0


def test_compute_answer_relevance_mock(evaluator: HallucinationEvaluator, mock_embedding_manager):
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."

    score = evaluator.compute_answer_relevance(
        question=question,
        answer=answer,
        embedding_manager=mock_embedding_manager,
    )
    assert 0.0 <= score <= 1.0


def test_compute_context_precision_mock(evaluator: HallucinationEvaluator, mock_embedding_manager):
    question = "Explain gradient descent optimization."
    chunks = [
        "Gradient descent iteratively updates parameters in the direction of the negative gradient.",
        "Stochastic gradient descent computes gradients on mini-batches.",
        "Learning rate controls step size along the gradient vector.",
    ]

    score = evaluator.compute_context_precision(
        question=question,
        chunks=chunks,
        embedding_manager=mock_embedding_manager,
    )
    assert 0.0 <= score <= 1.0


def test_compute_retrieval_precision_mock(evaluator: HallucinationEvaluator, mock_embedding_manager):
    question = "Quantum entanglement physics."
    chunks = ["Quantum entanglement occurs when particles remain connected.", "Irrelevant bakery recipe."]

    score = evaluator.compute_retrieval_precision(
        chunks=chunks,
        question=question,
        threshold=0.2,
        embedding_manager=mock_embedding_manager,
    )
    assert 0.0 <= score <= 1.0


def test_compute_answer_correctness(evaluator: HallucinationEvaluator):
    reference = "Photosynthesis converts light energy into chemical energy in green plants."
    answer = "In plants, photosynthesis transforms light into chemical energy."

    scores = evaluator.compute_answer_correctness(answer=answer, reference=reference)
    assert "rouge_l_f1" in scores
    assert "bertscore_f1" in scores
    assert "answer_correctness" in scores
    assert 0.0 <= scores["rouge_l_f1"] <= 1.0
    assert 0.0 <= scores["bertscore_f1"] <= 1.0
    assert 0.0 <= scores["answer_correctness"] <= 1.0


def test_evaluate_qa_pair(evaluator: HallucinationEvaluator):
    question = "What is reinforcement learning from human feedback?"
    answer = "RLHF aligns language model outputs with human preferences using reward modeling."
    context = [
        "Reinforcement learning from human feedback (RLHF) trains reward models to align LLMs with human values."
    ]
    reference = "RLHF aligns models with human intent using preference datasets and reward models."

    result = evaluator.evaluate_qa_pair(
        question=question,
        answer=answer,
        context_chunks=context,
        reference=reference,
        retrieved_ids=["doc_1"],
        relevant_ids=["doc_1"],
    )

    assert "metrics" in result
    assert "details" in result
    m = result["metrics"]
    assert "faithfulness" in m
    assert "hallucination_rate" in m
    assert "answer_relevance" in m
    assert "context_precision" in m
    assert "retrieval_precision" in m
    assert "answer_correctness" in m
    assert "retrieval_recall" in m
    assert m["retrieval_recall"] == 1.0
