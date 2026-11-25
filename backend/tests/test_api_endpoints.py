"""Exhaustive integration tests covering every single backend API endpoint."""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

# Initialize database schema for test runner
init_db()

client = TestClient(app)


def test_health_endpoints():
    """Verify both /health and /api/health return 200 OK."""
    r1 = client.get("/health")
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["status"] == "healthy"

    r2 = client.get("/api/health")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] == "healthy"


def test_settings_endpoints():
    """Verify GET and PATCH /api/settings with alias fields."""
    r_get = client.get("/api/settings")
    assert r_get.status_code == 200
    data = r_get.json()
    assert "default_llm_backend" in data
    assert "llm_backend" in data
    assert "chunk_size" in data

    # Patch with frontend alias keys
    r_patch = client.patch(
        "/api/settings",
        json={
            "top_k": 7,
            "chunk_size": 256,
            "chunk_overlap": 64,
            "llm_backend": "simulation",
        },
    )
    assert r_patch.status_code == 200
    p_data = r_patch.json()
    assert p_data["default_top_k"] == 7
    assert p_data["top_k"] == 7
    assert p_data["default_chunk_size"] == 256
    assert p_data["chunk_size"] == 256


def test_documents_lifecycle():
    """Verify document upload, chunk pagination, retrieval, and deletion."""
    test_content = (
        "Retrieval-Augmented Generation (RAG) significantly mitigates large language model hallucinations. "
        "Dense vector retrieval maps queries to embedding spaces using bi-encoders. "
        "Cross-encoders compute fine-grained relevance by joint attention over query and candidate chunks."
    )
    file_bytes = io.BytesIO(test_content.encode("utf-8"))

    # Upload
    upload_res = client.post(
        "/api/documents/upload",
        files={"file": ("rag_study.txt", file_bytes, "text/plain")},
        data={"description": "Test document for full system integration"},
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["id"]
    assert doc_id is not None
    assert upload_data["num_chunks"] >= 1

    # List
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["id"] == doc_id for d in docs)

    # Get Single
    get_res = client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["filename"] == "rag_study.txt"

    # Chunks with page/page_size
    chunks_paged = client.get(f"/api/documents/{doc_id}/chunks?page=1&page_size=10")
    assert chunks_paged.status_code == 200
    assert len(chunks_paged.json()["chunks"]) >= 1

    # Chunks with skip/limit
    chunks_skipped = client.get(f"/api/documents/{doc_id}/chunks?skip=0&limit=5")
    assert chunks_skipped.status_code == 200
    assert len(chunks_skipped.json()["chunks"]) >= 1

    # Search against uploaded document
    search_res = client.post(
        "/api/retrieval/search",
        json={"query": "What does RAG mitigate?", "top_k": 3, "mode": "semantic"},
    )
    assert search_res.status_code == 200
    assert len(search_res.json()["chunks"]) >= 1

    # Retrieval stats
    stats_res = client.get("/api/retrieval/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_chunks"] >= 1

    # Cleanup: Delete Document
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200


def test_qa_pipeline_and_sessions():
    """Verify QA answering with frontend aliases and session retrieval."""
    qa_res = client.post(
        "/api/qa/ask",
        json={
            "question": "What is Retrieval-Augmented Generation?",
            "top_k": 3,
            "retrieval_mode": "semantic",
            "use_reranking": False,
            "llm_backend": "simulation",
            "llm_model": "llama3-8b-8192",
            "run_standard": True,
            "run_rag": True,
        },
    )
    assert qa_res.status_code == 200
    qa_data = qa_res.json()
    session_id = qa_data["session_id"]
    assert session_id is not None
    assert qa_data["rag_answer"] is not None
    assert qa_data["standard_answer"] is not None

    # List Sessions
    sessions_res = client.get("/api/qa/sessions?limit=10")
    assert sessions_res.status_code == 200
    sessions = sessions_res.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)

    # Get Single Session
    s_res = client.get(f"/api/qa/sessions/{session_id}")
    assert s_res.status_code == 200
    assert s_res.json()["question"] == "What is Retrieval-Augmented Generation?"


def test_evaluation_pipeline():
    """Verify answer evaluation and metric persistence."""
    # Definitions
    defs_res = client.get("/api/evaluation/metrics/definitions")
    assert defs_res.status_code == 200
    assert "faithfulness" in defs_res.json()
    assert "hallucination_rate" in defs_res.json()

    # Evaluate
    eval_res = client.post(
        "/api/evaluation/evaluate",
        json={
            "question": "What improves RAG retrieval accuracy?",
            "answer": "Hybrid retrieval combining BM25 lexical search and dense bi-encoder embeddings improves context precision.",
            "context_chunks": [
                "Hybrid retrieval combining BM25 lexical search and dense bi-encoder embeddings improves context precision to 1.0."
            ],
            "reference_answer": "Hybrid retrieval combines BM25 and dense embeddings to improve accuracy.",
        },
    )
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "faithfulness" in eval_data["metrics"]
    assert eval_data["metrics"]["faithfulness"] >= 0.8
    assert eval_data["metrics"]["hallucination_rate"] <= 0.2

    # Results list
    res_list = client.get("/api/evaluation/results?limit=10")
    assert res_list.status_code == 200
    assert res_list.json()["total"] >= 1


def test_experiments_pipeline():
    """Verify benchmark listing, experiment execution with flat config, and retrieval."""
    # Benchmarks list
    b_list = client.get("/api/experiments/benchmarks/list")
    assert b_list.status_code == 200
    assert len(b_list.json()) >= 1

    # Run experiment with flat config
    exp_res = client.post(
        "/api/experiments/run",
        json={
            "name": "Quick Hyperparameter Test",
            "description": "Verifies flat config parsing and runs generation",
            "variable": "top_k",
            "values": [1, 2],
            "questions": ["What is RAG?"],
            "reference_answers": ["RAG stands for retrieval augmented generation."],
            "base_config": {
                "top_k": 2,
                "retrieval_mode": "semantic",
                "use_reranking": False,
                "llm_backend": "simulation",
                "llm_model": "llama3-8b-8192",
            },
        },
    )
    assert exp_res.status_code == 201
    exp_data = exp_res.json()
    exp_id = exp_data["id"]
    assert exp_data["status"] == "completed"
    assert exp_data["runs"] is not None
    assert len(exp_data["runs"]) >= 2

    # List experiments
    list_res = client.get("/api/experiments")
    assert list_res.status_code == 200
    assert any(e["id"] == exp_id for e in list_res.json()["experiments"])

    # Get single experiment
    get_res = client.get(f"/api/experiments/{exp_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Quick Hyperparameter Test"
    assert len(get_res.json()["runs"]) >= 2

    # Delete experiment
    del_res = client.delete(f"/api/experiments/{exp_id}")
    assert del_res.status_code == 200


def test_analytics_and_reports():
    """Verify analytics summaries, trends, taxonomy, and file exports."""
    # Summary
    summary_res = client.get("/api/analytics/summary")
    assert summary_res.status_code == 200
    assert "total_qa_sessions" in summary_res.json()

    # Model comparison
    model_res = client.get("/api/analytics/model-comparison")
    assert model_res.status_code == 200
    assert "models" in model_res.json()

    # Time series
    ts_res = client.get("/api/analytics/metrics-over-time?metric_name=faithfulness&limit=20")
    assert ts_res.status_code == 200
    assert ts_res.json()["metric_name"] == "faithfulness"

    # Hallucination trend
    trend_res = client.get("/api/analytics/hallucination-trend?limit=20")
    assert trend_res.status_code == 200

    # Taxonomy
    tax_res = client.get("/api/analytics/hallucination-taxonomy")
    assert tax_res.status_code == 200
    assert "taxonomy" in tax_res.json()

    # Research findings
    findings_res = client.get("/api/analytics/research-findings")
    assert findings_res.status_code == 200
    assert "answers" in findings_res.json()

    # CSV Export
    csv_res = client.get("/api/reports/export/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]

    # JSON Export
    json_res = client.get("/api/reports/export/json")
    assert json_res.status_code == 200
    assert "documents" in json_res.json()

    # Text Report
    text_res = client.get("/api/reports/report/text")
    assert text_res.status_code == 200
    assert "Empirical Research Report" in text_res.text
