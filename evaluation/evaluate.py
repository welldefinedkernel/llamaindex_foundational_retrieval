import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

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
    model: str = "gpt-5.4-mini",
    threshold: float = 0.5,
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
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
        ContextualRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        ),
    ]

    return evaluate(test_cases=test_cases, metrics=metrics)