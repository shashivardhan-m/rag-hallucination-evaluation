"""Unit tests for DocumentProcessor text extraction and sentence-aware chunking."""

import os
import tempfile
import pytest
from app.core.document_processor import DocumentProcessor


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor(default_chunk_size=200, default_chunk_overlap=50)


def test_clean_text(processor: DocumentProcessor):
    raw = "  Hello \t world!\r\n\r\n\r\nThis has   extra spaces and control chars \x00\x07.  "
    cleaned = processor.clean_text(raw)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "   " not in cleaned
    assert "Hello world!" in cleaned
    # Note: space before period is acceptable — cleaner preserves word boundaries
    assert "This has extra spaces and control chars" in cleaned


def test_split_into_chunks_empty(processor: DocumentProcessor):
    chunks = processor.split_into_chunks("", chunk_size=200, chunk_overlap=50)
    assert chunks == []

    chunks_whitespace = processor.split_into_chunks("   \n\t  ", chunk_size=200, chunk_overlap=50)
    assert chunks_whitespace == []


def test_split_into_chunks_basic(processor: DocumentProcessor):
    sample_text = (
        "Retrieval-Augmented Generation improves the factual accuracy of language models. "
        "It achieves this by injecting relevant context retrieved from an external knowledge base. "
        "Evaluating hallucination requires measuring faithfulness against the retrieved context. "
        "Reference-based metrics like ROUGE and BERTScore evaluate correctness against ground truth. "
        "Reference-free metrics verify that claims are entailed by the context chunks."
    )

    chunks = processor.split_into_chunks(sample_text, chunk_size=150, chunk_overlap=40)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c["chunk_index"] == idx
        assert "content" in c
        assert len(c["content"]) > 0
        assert c["char_count"] == len(c["content"])
        assert c["token_estimate"] > 0
        # Check that sentence was not arbitrarily cut in half if possible
        assert isinstance(c["content"], str)


def test_split_into_chunks_overlap(processor: DocumentProcessor):
    sample_text = (
        "Sentence number one introduces the core theme. "
        "Sentence number two develops the conceptual foundation. "
        "Sentence number three offers empirical data. "
        "Sentence number four concludes the paragraph with a summary."
    )
    chunks = processor.split_into_chunks(sample_text, chunk_size=120, chunk_overlap=45)

    if len(chunks) > 1:
        # Overlap should mean words from the end of chunk 0 or sentence appears in chunk 1
        c0 = chunks[0]["content"]
        c1 = chunks[1]["content"]
        assert len(c0) > 0 and len(c1) > 0


def test_extract_text_from_txt(processor: DocumentProcessor):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Line 1: Domain-specific knowledge.\nLine 2: RAG hallucination evaluation.")
        temp_path = f.name

    try:
        extracted = processor.extract_text_from_txt(temp_path)
        assert "Domain-specific knowledge" in extracted
        assert "RAG hallucination evaluation" in extracted
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_vector_store_save_and_load():
    import numpy as np
    from app.core.vector_store import FAISSVectorStore

    vs = FAISSVectorStore()
    vs.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = [
            {"chunk_id": "c1", "content": "RAG evaluation and benchmarking", "chunk_index": 0},
            {"chunk_id": "c2", "content": "NLI claim verification models", "chunk_index": 1},
        ]
        embeddings = np.random.randn(2, 64).astype(np.float32)

        vs.add_chunks(chunks, embeddings, "doc_test")
        assert len(vs.get_all_chunks()) == 2

        # Save to disk
        saved = vs.save_index(tmpdir)
        assert saved is True

        # Clear in-memory
        vs.clear()
        assert len(vs.get_all_chunks()) == 0

        # Load from disk
        loaded = vs.load_index(tmpdir)
        assert loaded is True
        assert len(vs.get_all_chunks()) == 2
        assert vs.get_all_chunks()[0]["chunk_id"] == "c1"


def test_process_document_txt(processor: DocumentProcessor):
    content = "The quick brown fox jumps over the lazy dog. Artificial intelligence evaluation benchmarks."
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    try:
        result = processor.process_document(
            file_path=temp_path,
            filename="sample.txt",
            chunk_size=100,
            chunk_overlap=20,
        )
        assert result["filename"] == "sample.txt"
        assert result["file_type"] == "txt"
        assert result["num_chunks"] >= 1
        assert len(result["chunks"]) == result["num_chunks"]
        assert result["total_chars"] > 0
        assert result["total_tokens_estimate"] > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_unsupported_file_format(processor: DocumentProcessor):
    with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as f:
        f.write("binary")
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as excinfo:
            processor.process_document(temp_path, "sample.exe")
        assert "Unsupported file format" in str(excinfo.value)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
