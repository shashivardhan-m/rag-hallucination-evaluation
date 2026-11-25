"""Standard Hallucination Evaluation Benchmarks Loader (RAGTruth & HaluEval)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkItem(BaseModel):
    """Schema for a standardized benchmark evaluation item."""
    id: str
    benchmark: str
    question: str
    gold_context: List[str]
    ground_truth_answer: str
    hallucinated_answer: Optional[str] = None
    distractor_context: List[str] = Field(default_factory=list)
    irrelevant_context: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    task: Optional[str] = None
    hallucination_types: List[str] = Field(default_factory=list)


class BenchmarkRegistry:
    """Registry and loader for standard hallucination benchmarks."""

    _instance: Optional["BenchmarkRegistry"] = None

    def __init__(self, benchmark_dir: Optional[Path] = None) -> None:
        if benchmark_dir is None:
            # Default to backend/data/benchmarks
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.benchmark_dir = base_dir / "data" / "benchmarks"
        else:
            self.benchmark_dir = Path(benchmark_dir)

    @classmethod
    def get_instance(cls) -> "BenchmarkRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all available benchmark datasets and metadata."""
        benchmarks = []
        ragtruth_path = self.benchmark_dir / "ragtruth_benchmark.json"
        halueval_path = self.benchmark_dir / "halueval_benchmark.json"

        if ragtruth_path.exists():
            with open(ragtruth_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            benchmarks.append({
                "id": "ragtruth",
                "name": "RAGTruth Benchmark",
                "description": "Standardized benchmark for evaluating claim-level hallucinations in RAG systems across technical domains.",
                "total_items": len(items),
                "domains": list(set(it.get("domain", "") for it in items if it.get("domain"))),
                "tasks": ["Question Answering", "Knowledge Extraction"],
                "file_path": str(ragtruth_path),
            })

        if halueval_path.exists():
            with open(halueval_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            benchmarks.append({
                "id": "halueval",
                "name": "HaluEval Benchmark",
                "description": "Large-scale benchmark evaluating LLM and RAG factual consistency, hallucinated completions, and distractor resistance.",
                "total_items": len(items),
                "domains": ["Science", "Computer Science", "Biomedicine"],
                "tasks": list(set(it.get("task", "") for it in items if it.get("task"))),
                "file_path": str(halueval_path),
            })

        return benchmarks

    def load_benchmark(self, benchmark_id: str) -> List[BenchmarkItem]:
        """Load benchmark items by ID ('ragtruth', 'halueval', or 'all')."""
        normalized = benchmark_id.lower().strip()
        items: List[BenchmarkItem] = []

        if normalized in ("ragtruth", "all"):
            p = self.benchmark_dir / "ragtruth_benchmark.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    items.extend([BenchmarkItem(**it) for it in raw])

        if normalized in ("halueval", "all"):
            p = self.benchmark_dir / "halueval_benchmark.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    items.extend([BenchmarkItem(**it) for it in raw])

        if not items and normalized not in ("ragtruth", "halueval", "all"):
            raise ValueError(f"Unknown benchmark '{benchmark_id}'. Available: 'ragtruth', 'halueval', 'all'")

        return items

    def get_perturbed_contexts(
        self,
        item: BenchmarkItem,
        context_condition: str = "gold",
    ) -> List[str]:
        """
        Produce context chunks according to the experimental perturbation condition:
        - 'gold': strictly ground-truth relevant passages
        - 'distractor': gold passages mixed with semantically adjacent distractor passages
        - 'irrelevant': completely out-of-domain irrelevant passages (tests abstention)
        - 'none': empty context list (tests unaugmented parametric generation)
        """
        cond = context_condition.lower().strip()
        if cond == "gold":
            return list(item.gold_context)
        elif cond == "distractor":
            # Gold + Distractor mixed
            return list(item.gold_context) + list(item.distractor_context)
        elif cond == "irrelevant":
            return list(item.irrelevant_context)
        elif cond in ("none", "parametric", "empty"):
            return []
        else:
            return list(item.gold_context)

    def export_as_documents(self, benchmark_id: str = "all") -> List[Dict[str, Any]]:
        """
        Convert benchmark passages into standard document chunk dictionaries
        suitable for direct indexing in the FAISS / BM25 retriever.
        """
        items = self.load_benchmark(benchmark_id)
        docs = []
        for it in items:
            # Gold chunks
            for idx, chunk in enumerate(it.gold_context):
                docs.append({
                    "id": f"{it.id}_gold_{idx}",
                    "document_id": f"benchmark_{it.benchmark.lower()}",
                    "filename": f"{it.benchmark.lower()}_{it.id}.txt",
                    "chunk_index": idx,
                    "content": chunk,
                    "is_gold": True,
                    "target_question": it.question,
                })
            # Distractor chunks
            for idx, chunk in enumerate(it.distractor_context):
                docs.append({
                    "id": f"{it.id}_distractor_{idx}",
                    "document_id": f"benchmark_{it.benchmark.lower()}",
                    "filename": f"{it.benchmark.lower()}_{it.id}_distractor.txt",
                    "chunk_index": idx + 100,
                    "content": chunk,
                    "is_gold": False,
                    "target_question": it.question,
                })
            # Irrelevant chunks
            for idx, chunk in enumerate(it.irrelevant_context):
                docs.append({
                    "id": f"{it.id}_irrelevant_{idx}",
                    "document_id": f"benchmark_{it.benchmark.lower()}",
                    "filename": f"{it.benchmark.lower()}_{it.id}_irrelevant.txt",
                    "chunk_index": idx + 200,
                    "content": chunk,
                    "is_gold": False,
                    "target_question": it.question,
                })
        return docs


def get_benchmark_registry() -> BenchmarkRegistry:
    """Return singleton instance of BenchmarkRegistry."""
    return BenchmarkRegistry.get_instance()
