"""SQLAlchemy database models for documents, chunks, QA sessions, evaluations, and experiments."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from app.database import Base


def generate_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


def get_current_time() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class Document(Base):
    """Uploaded document record."""

    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    num_chunks = Column(Integer, default=0, nullable=False)
    upload_time = Column(DateTime, default=get_current_time, nullable=False)
    description = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True, default="{}")

    # Relationships
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )


class DocumentChunk(Base):
    """Text chunk extracted from an uploaded document."""

    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    char_count = Column(Integer, default=0, nullable=False)
    token_estimate = Column(Integer, default=0, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class QASession(Base):
    """Session comparing standard LLM generation and RAG generation."""

    __tablename__ = "qa_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    question = Column(Text, nullable=False)
    standard_answer = Column(Text, nullable=True)
    rag_answer = Column(Text, nullable=True)
    standard_latency_ms = Column(Float, default=0.0, nullable=False)
    rag_latency_ms = Column(Float, default=0.0, nullable=False)
    top_k = Column(Integer, default=5, nullable=False)
    model_used = Column(String(100), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    session_time = Column(DateTime, default=get_current_time, nullable=False)
    documents_used = Column(Text, nullable=True, default="[]")  # JSON list of doc IDs or titles

    # Relationships
    evaluations = relationship(
        "EvaluationResult",
        back_populates="qa_session",
        cascade="all, delete-orphan",
    )


class EvaluationResult(Base):
    """Evaluation metrics computed on answers."""

    __tablename__ = "evaluation_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    qa_session_id = Column(
        String(36),
        ForeignKey("qa_sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    is_reference_based = Column(Boolean, default=False, nullable=False)
    computed_at = Column(DateTime, default=get_current_time, nullable=False)
    notes = Column(Text, nullable=True)

    # Relationships
    qa_session = relationship("QASession", back_populates="evaluations")


class Experiment(Base):
    """Experiment comparing RAG configurations across test queries."""

    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    config_json = Column(Text, nullable=False)  # JSON representation of ExperimentConfig
    results_json = Column(Text, nullable=True)  # Detailed benchmark results JSON
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    created_at = Column(DateTime, default=get_current_time, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    num_queries = Column(Integer, default=0, nullable=False)
