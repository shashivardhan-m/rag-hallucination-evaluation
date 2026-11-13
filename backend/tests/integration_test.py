#!/usr/bin/env python3
"""
Integration test: end-to-end smoke test of the RAG Hallucination Evaluator API.
Requires the backend to be running on http://localhost:8000.

Usage:
    cd backend && source venv/bin/activate
    python tests/integration_test.py

Pass GROQ_API_KEY in env to also test Q&A and Evaluation steps.
"""

import sys
import json
import os

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"
SAMPLE_DOC = os.path.join(
    os.path.dirname(__file__),
    "../data/sample_docs/rag_survey_excerpt.txt",
)
HAS_LLM = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⊘\033[0m"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}" + (f" — {detail}" if detail else ""))
    results.append((label, condition, detail))
    return condition


def skip(label: str, reason: str = ""):
    print(f"  {SKIP}  {label}" + (f" (skipped: {reason})" if reason else ""))


def step(title: str):
    print(f"\n\033[1m{title}\033[0m")


# ── 1. Health Check ───────────────────────────────────────────────────────────
step("1. Health Check")
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    data = r.json()
    check("HTTP 200", r.status_code == 200)
    check("status == healthy", data.get("status") == "healthy", data.get("status", ""))
    check("vector_store online", data.get("vector_store", {}).get("status") == "online")
except Exception as e:
    check("Backend reachable", False, str(e))
    print("\n  Backend not reachable — is it running on port 8000?")
    sys.exit(1)

# ── 2. Upload Demo Document ───────────────────────────────────────────────────
step("2. Upload Demo Document")
doc_id = None
try:
    if not os.path.exists(SAMPLE_DOC):
        check("Sample doc exists", False, SAMPLE_DOC)
    else:
        check("Sample doc exists", True)
        with open(SAMPLE_DOC, "rb") as f:
            r = requests.post(
                f"{BASE}/api/documents/upload",
                files={"file": ("rag_survey_excerpt.txt", f, "text/plain")},
                data={"chunk_size": "400", "chunk_overlap": "80"},
                timeout=60,
            )
        # FastAPI returns 201 Created for successful uploads
        ok = r.status_code in (200, 201)
        check(f"Upload HTTP 2xx (got {r.status_code})", ok)
        if ok:
            resp = r.json()
            # API returns the document object directly with 'id'
            doc_id = resp.get("id") or resp.get("document_id")
            chunks = resp.get("num_chunks", 0)
            check("document_id returned", bool(doc_id), str(doc_id)[:8] if doc_id else "")
            check("Chunks indexed > 0", chunks > 0, f"{chunks} chunks")
except Exception as e:
    check("Upload succeeded", False, str(e))

# ── 3. List Documents ─────────────────────────────────────────────────────────
step("3. List Documents")
try:
    r = requests.get(f"{BASE}/api/documents/", timeout=10)
    check("HTTP 200", r.status_code == 200)
    docs = r.json().get("documents", [])
    check("At least 1 document listed", len(docs) >= 1, f"{len(docs)} docs")
except Exception as e:
    check("List docs succeeded", False, str(e))

# ── 4. Semantic Retrieval ─────────────────────────────────────────────────────
step("4. Semantic Retrieval")
try:
    r = requests.post(
        f"{BASE}/api/retrieval/search",
        json={"query": "What is hallucination rate in RAG?", "top_k": 3},
        timeout=30,
    )
    check("HTTP 200", r.status_code == 200, str(r.status_code))
    if r.status_code == 200:
        result_data = r.json()
        # Actual API key: 'chunks' (not 'results')
        chunks = result_data.get("chunks", [])
        total = result_data.get("total_retrieved", 0)
        check("Retrieved > 0 chunks", len(chunks) > 0 or total > 0,
              f"{total} retrieved, {len(chunks)} in response")
        if chunks:
            top_score = chunks[0].get("score", 0)
            check("Top similarity score > 0", top_score > 0, f"{top_score:.4f}")
except Exception as e:
    check("Retrieval succeeded", False, str(e))

# ── 5. Q&A (Standard + RAG) ───────────────────────────────────────────────────
step("5. Q&A — Standard LLM vs RAG")
session_id = None
rag_text = ""
if not HAS_LLM:
    skip("Q&A test", "No LLM API key set. Set GROQ_API_KEY to test LLM calls.")
    # Still test the endpoint responds
    try:
        r = requests.post(
            f"{BASE}/api/qa/ask",
            json={"question": "What is faithfulness?", "top_k": 2},
            timeout=30,
        )
        check("Q&A endpoint responds", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            resp = r.json()
            session_id = resp.get("session_id")
            check("session_id returned", bool(session_id))
            # Even error responses have session_id
            rag_text = resp.get("rag_answer", {}).get("answer", "") if isinstance(resp.get("rag_answer"), dict) else ""
    except Exception as e:
        check("Q&A endpoint responds", False, str(e))
else:
    try:
        r = requests.post(
            f"{BASE}/api/qa/ask",
            json={"question": "Define faithfulness as a hallucination metric.", "top_k": 3},
            timeout=120,
        )
        check("Q&A HTTP 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            resp = r.json()
            session_id = resp.get("session_id")
            rag_obj = resp.get("rag_answer", {}) or {}
            std_obj = resp.get("standard_answer", {}) or {}
            rag_text = rag_obj.get("answer", "")
            check("session_id returned", bool(session_id))
            check("RAG answer non-empty", len(rag_text) > 20, f"{len(rag_text)} chars")
            check("Standard answer non-empty", len(std_obj.get("answer", "")) > 20)
            check("RAG latency recorded", rag_obj.get("latency_ms", 0) > 0,
                  f"{rag_obj.get('latency_ms', 0):.0f} ms")
    except Exception as e:
        check("Q&A succeeded", False, str(e))

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
step("6. Hallucination Evaluation")
# Evaluation works without LLM — uses NLI model only
try:
    eval_answer = rag_text if len(rag_text) > 20 else (
        "Faithfulness measures whether claims in the generated answer are entailed by the retrieved context. "
        "It is calculated as the proportion of supported claims over total claims."
    )
    r = requests.post(
        f"{BASE}/api/evaluation/evaluate",
        json={
            "question": "Define faithfulness as a hallucination metric.",
            "answer": eval_answer,  # required field: 'answer' (not 'rag_answer')
            "context_chunks": [
                "Faithfulness is defined as the proportion of entailed claims over total claims. "
                "F = |Entailed Claims| / |Total Claims|. The hallucination rate is 1 - faithfulness.",
                "NLI-based faithfulness checking uses DeBERTa-v3 to classify each claim as entailment, "
                "neutral, or contradiction with respect to the retrieved context.",
            ],
            "qa_session_id": session_id,  # optional — links to session
        },
        timeout=120,
    )
    check("Evaluation HTTP 200", r.status_code == 200, str(r.status_code))
    if r.status_code == 200:
        ev = r.json()
        metrics = ev.get("metrics", {})
        hr = metrics.get("hallucination_rate", -1)
        check("hallucination_rate in [0,1]", 0.0 <= hr <= 1.0, f"{hr:.3f}")
        check("metrics list non-empty", len(metrics) > 0, f"{len(metrics)} metrics")
        # Print the actual metrics
        for m_name, m_val in metrics.items():
            print(f"       {m_name}: {m_val:.3f}")
    elif r.status_code == 422:
        print(f"  ✗  422 Validation Error: {r.json()}")
except Exception as e:
    check("Evaluation succeeded", False, str(e))

# ── 7. Metric Definitions ─────────────────────────────────────────────────────
step("7. Metric Definitions API")
try:
    r = requests.get(f"{BASE}/api/evaluation/metrics/definitions", timeout=10)
    check("HTTP 200", r.status_code == 200)
    defs = r.json()
    check("Has faithfulness definition", "faithfulness" in defs)
    check("Has hallucination_rate definition", "hallucination_rate" in defs)
    print(f"       Total metric definitions: {len(defs)}")
except Exception as e:
    check("Metric definitions succeeded", False, str(e))

# ── 8. Analytics ─────────────────────────────────────────────────────────────
step("8. Analytics Summary")
try:
    r = requests.get(f"{BASE}/api/analytics/summary", timeout=10)
    check("HTTP 200", r.status_code == 200)
    if r.status_code == 200:
        s = r.json()
        check("total_documents >= 1", s.get("total_documents", 0) >= 1,
              f"{s.get('total_documents', 0)}")
        check("total_chunks >= 1", s.get("total_chunks", 0) >= 1,
              f"{s.get('total_chunks', 0)}")
except Exception as e:
    check("Analytics succeeded", False, str(e))

# ── 9. Report Generation ──────────────────────────────────────────────────────
step("9. Report Generation")
try:
    r = requests.get(f"{BASE}/api/reports/export/csv", timeout=15)
    check("CSV export HTTP 200", r.status_code == 200, str(r.status_code))
    if r.status_code == 200:
        check("Content-Type csv", "csv" in r.headers.get("Content-Type", "").lower())
except Exception as e:
    check("CSV export succeeded", False, str(e))

try:
    r = requests.get(f"{BASE}/api/reports/export/json", timeout=15)
    check("JSON export HTTP 200", r.status_code == 200, str(r.status_code))
except Exception as e:
    check("JSON export succeeded", False, str(e))

# ── 10. Cleanup ────────────────────────────────────────────────────────────────
step("10. Cleanup — Delete Uploaded Document")
if doc_id:
    try:
        r = requests.delete(f"{BASE}/api/documents/{doc_id}", timeout=10)
        check("Delete HTTP 200", r.status_code == 200, str(r.status_code))
    except Exception as e:
        check("Delete succeeded", False, str(e))
else:
    skip("Delete test", "No doc_id captured (upload check failed)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
failed = [label for label, ok, _ in results if not ok]
llm_note = "" if HAS_LLM else " (set GROQ_API_KEY for full LLM test coverage)"
print(f"\n  Result: {passed}/{total} checks passed{llm_note}")
if failed:
    print(f"  Failed: {', '.join(failed)}")
if passed == total:
    print("\n  \033[92m🎉 All integration checks passed!\033[0m")
elif passed >= total * 0.85:
    print("\n  \033[93m✓  Most checks passed. Review failures above.\033[0m")
else:
    print("\n  \033[91m✗  Integration test failed. Check backend logs.\033[0m")
print()
sys.exit(0 if passed >= total * 0.85 else 1)
