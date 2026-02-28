from __future__ import annotations

import csv
from pathlib import Path

from anonlm.benchmarking.runner import run_benchmark
from anonlm.benchmarking.splits import BenchmarkSplit
from anonlm.config import AnonLMConfig


class FakeEngine:
    def __init__(self) -> None:
        self.config = AnonLMConfig(api_key="test-key")

    def detect_entities(self, text: str):
        if "Jane Doe" in text:
            return [{"type": "PERSON", "text": "Jane Doe"}]
        return []


def test_run_benchmark_without_history(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    with open(dataset, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "chunk_id", "input", "expected"])
        writer.writerow(
            ["DOC_001", "1", "Jane Doe arrived.", '[{"type": "PERSON", "text": "Jane Doe"}]']
        )
        writer.writerow(["DOC_002", "1", "No pii.", "[]"])

    result = run_benchmark(
        engine=FakeEngine(),
        dataset_path=str(dataset),
        split=BenchmarkSplit.DEV,
        save_history=False,
        stream=None,
    )

    assert result.exit_code == 0
    assert result.artifact_path is None
    assert int(result.overall["tp"]) >= 1
