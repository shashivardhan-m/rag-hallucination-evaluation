"""RAG Pipeline orchestrating context retrieval and LLM prompt generation."""

import time
from typing import Any, Dict, Optional

from app.config import get_settings
from app.core.embeddings import EmbeddingManager, get_embedding_manager
from app.core.llm_client import LLMClient, get_llm_client
from app.core.retriever import Retriever, get_retriever
from app.models.schemas import AnswerResult
from app.utils.text_utils import format_context_for_prompt

STANDARD_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant. "
    "Answer the user's question directly, clearly, and concisely based on your general knowledge."
)

RAG_SYSTEM_PROMPT = (
    "You are a factual, evidence-grounded AI assistant.\n"
    "Answer the user's question ONLY using the facts explicitly stated in the context provided below.\n"
    "Do not extrapolate, invent, or use outside training assumptions.\n"
    "If the provided context does not contain sufficient information to answer the question, explicitly state:\n"
    "'I cannot find sufficient information in the provided documents to answer this question.'\n"
    "Always cite the relevant source documents and chunks in your answer."
)


class RAGPipeline:
    """Executes standard LLM and grounded RAG answer generation pipelines."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever or get_retriever()
        self.llm_client = llm_client or get_llm_client()
        self.embedding_manager = embedding_manager or get_embedding_manager()

    def _resolve_client(self, model_config: Optional[Dict[str, Any]] = None) -> LLMClient:
        """Instantiate or reuse client based on per-request overrides."""
        if not model_config:
            return self.llm_client

        backend = model_config.get("backend") or self.llm_client.backend
        model = model_config.get("model") or self.llm_client.model
        api_key = model_config.get("api_key") or self.llm_client.api_key
        base_url = model_config.get("base_url") or self.llm_client.base_url

        return LLMClient(backend=backend, model=model, api_key=api_key, base_url=base_url)

    def answer_standard(
        self,
        question: str,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> AnswerResult:
        """Answer a question directly using general LLM knowledge (baseline without retrieval)."""
        client = self._resolve_client(model_config)
        cfg = model_config or {}
        temperature = float(cfg.get("temperature", 0.1))
        max_tokens = int(cfg.get("max_tokens", 1024))

        answer, latency_ms = client.generate(
            prompt=question,
            system_prompt=STANDARD_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return AnswerResult(
            answer=answer,
            latency_ms=latency_ms,
            model=client.model,
            backend=client.backend,
            retrieved_chunks=None,
        )

    def answer_with_rag(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "semantic",
        use_reranking: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> AnswerResult:
        """Answer a question strictly grounded on retrieved document chunks."""
        overall_start = time.perf_counter()

        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            mode=retrieval_mode,
            use_reranking=use_reranking,
            document_id=document_id,
        )

        context_str = format_context_for_prompt(chunks)

        rag_prompt = (
            f"Here is the verified context from our document store:\n"
            f"----------------------------------------\n"
            f"{context_str}\n"
            f"----------------------------------------\n\n"
            f"User Question: {question}\n\n"
            f"Provide your answer grounded strictly in the context above:"
        )

        client = self._resolve_client(model_config)
        cfg = model_config or {}
        temperature = float(cfg.get("temperature", 0.0))
        max_tokens = int(cfg.get("max_tokens", 1024))

        answer, _ = client.generate(
            prompt=rag_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_latency_ms = (time.perf_counter() - overall_start) * 1000.0

        return AnswerResult(
            answer=answer,
            latency_ms=round(total_latency_ms, 2),
            model=client.model,
            backend=client.backend,
            retrieved_chunks=chunks,
        )


def get_rag_pipeline() -> RAGPipeline:
    """Return an initialized RAGPipeline."""
    return RAGPipeline()
