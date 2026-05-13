import json
import numpy as np

from collections.abc import Callable
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pathlib import Path
from typing import Any

def _load_json_dataset(data_path: str) -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parent.parent
    data_path_resolved = Path(data_path) if Path(data_path).is_absolute() else project_root / data_path

    with data_path_resolved.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of samples in {data_path_resolved}")

    return payload


def build_retrieval_test_cases(
    data_path: str,
    retrieval_context_getter: Callable[[str], list[str]] | None = None,
) -> list[LLMTestCase]:
    """Create DeepEval test cases for retrieval quality metrics.

    Dataset schema expected per sample:
    - input: str
    - expected_output: str
    - actual_output: str | None
    - context: list[str] (ground truth supporting chunks)
    - retrieval_context: list[str] | None (chunks returned by retriever)
    """

    samples = _load_json_dataset(data_path)
    test_cases: list[LLMTestCase] = []

    for sample in samples:
        input_text = sample.get("input", "")
        expected_output = sample.get("expected_output", "")
        actual_output = sample.get("actual_output") or ""

        context = sample.get("context") or []
        retrieval_context = sample.get("retrieval_context") or []

        if not retrieval_context and retrieval_context_getter is not None:
            retrieval_context = retrieval_context_getter(input_text)

        if not retrieval_context:
            continue

        test_cases.append(
            LLMTestCase(
                input=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                context=context,
                retrieval_context=retrieval_context,
            )
        )

    return test_cases


def evaluate_retrieval_metrics(
    data_path: str,
    retrieval_context_getter: Callable[[str], list[str]] | None = None,
    model: DeepEvalBaseLLM | str = "gpt-5.4-mini",
    threshold: float = 0.5,
    async_mode: bool = True,
    include_reason: bool = True,
):
    """Run retrieval-focused DeepEval metrics for the synthetic dataset."""
    test_cases = build_retrieval_test_cases(
        data_path=data_path,
        retrieval_context_getter=retrieval_context_getter,
    )

    if not test_cases:
        raise ValueError(
            "No test cases contain retrieval_context. "
            "Provide retrieval_context_getter or populate retrieval_context in the dataset."
        )

    metrics = [
        ContextualPrecisionMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
        ContextualRecallMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
        ContextualRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
        ),
    ]

    return evaluate(test_cases=test_cases, metrics=metrics)

def chunk_overlap(retrieved_chunk: str, ground_truth_chunk: str, threshold=0.3) -> bool:
    """Check if retrieved chunk overlaps with ground truth using token overlap."""
    ret_tokens = set(retrieved_chunk.lower().split())
    gt_tokens = set(ground_truth_chunk.lower().split())
    if not gt_tokens:
        return False
    overlap = len(ret_tokens & gt_tokens) / len(gt_tokens)
    return overlap >= threshold

def evaluate_retrieval(
    data_path: str,
    retrieval_context_getter: Callable[[str], list[str]],
    k: int = 5,
    threshold: float = 0.3,
) -> dict[str, float]:
    """Evaluate retrieval quality using automatic token-overlap metrics.
    
    This function uses automatic evaluation (token overlap checking) rather than 
    LLM-as-Judge. Retrieved chunks are compared against ground truth contexts 
    using word-level overlap to compute recall, precision, MRR, and F1 scores.
    
    Args:
        data_path: Path to JSON dataset with 'input', 'context', and optionally 'retrieval_context' fields.
        retrieval_context_getter: Callable to retrieve chunks for a given query.
        k: Number of top chunks to retrieve for each query.
        threshold: Minimum overlap ratio (0-1) to consider chunks as matching.
    
    Returns:
        Dictionary with 'recall', 'precision', 'mrr', 'f1' metrics.
    """
    samples = _load_json_dataset(data_path)
    
    recalls = []
    precisions = []
    reciprocal_ranks = []

    for sample in samples:
        query = sample.get("input", "")
        gt_contexts = sample.get("context") or []
        
        if not query or not gt_contexts:
            continue
        
        retrieved_chunks = retrieval_context_getter(query)[:k]
        
        if not retrieved_chunks:
            continue
        
        # Compute recall: fraction of ground truth chunks that have overlap with retrieval
        hits = sum(
            1 for gt_chunk in gt_contexts
            if any(chunk_overlap(ret, gt_chunk, threshold) for ret in retrieved_chunks)
        )
        recall = hits / len(gt_contexts) if gt_contexts else 0.0
        
        # Compute precision: fraction of retrieved chunks that match ground truth
        relevant_retrieved = sum(
            1 for ret in retrieved_chunks
            if any(chunk_overlap(ret, gt_chunk, threshold) for gt_chunk in gt_contexts)
        )
        precision = relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        # Compute reciprocal rank: 1/rank of first relevant chunk
        reciprocal_rank = 0.0
        for rank, ret in enumerate(retrieved_chunks, start=1):
            if any(chunk_overlap(ret, gt_chunk, threshold) for gt_chunk in gt_contexts):
                reciprocal_rank = 1.0 / rank
                break
        
        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(reciprocal_rank)
    
    mean_recall = np.mean(recalls) if recalls else 0.0
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    f1 = 2 * mean_recall * mean_precision / (mean_recall + mean_precision + 1e-9)
    
    print(f"Automatic Metric Evaluation (Token Overlap, threshold={threshold})")
    print(f"Recall@{k}:    {mean_recall:.3f} (std: {np.std(recalls):.3f})" if recalls else f"Recall@{k}:    N/A")
    print(f"Precision@{k}: {mean_precision:.3f} (std: {np.std(precisions):.3f})" if precisions else f"Precision@{k}: N/A")
    print(f"MRR@{k}:       {mean_reciprocal_rank:.3f} (std: {np.std(reciprocal_ranks):.3f})" if reciprocal_ranks else f"MRR@{k}:       N/A")
    print(f"F1@{k}:        {f1:.3f}")
    if recalls:
        print(f"\nPer-query recall distribution (n={len(recalls)}):")
        print(f"  100% recall: {sum(1 for r in recalls if r == 1.0)}")
        print(f"  >50% recall: {sum(1 for r in recalls if r > 0.5)}")
        print(f"  0% recall:   {sum(1 for r in recalls if r == 0.0)}")

    return {
        "recall": float(mean_recall),
        "precision": float(mean_precision),
        "mrr": float(mean_reciprocal_rank),
        "f1": float(f1),
    }