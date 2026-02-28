"""Benchmark execution pipeline."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from anonlm.benchmarking.history import write_run_artifacts
from anonlm.benchmarking.reporting import print_report
from anonlm.benchmarking.splits import BenchmarkSplit, select_rows_for_split
from anonlm.config import AnonLMConfig
from anonlm.engine import AnonymizationEngine
from anonlm.normalizer import normalize
from anonlm.prompts import SYSTEM_PROMPT
from anonlm.schema import PIIType


@dataclass(frozen=True)
class EntityKey:
    pii_type: str
    canonical: str


@dataclass
class RowResult:
    doc_id: str
    chunk_id: str
    input: str
    expected_keys: set[EntityKey]
    detected_keys: set[EntityKey]
    detected_raw: list[dict[str, str]]

    @property
    def tp(self) -> set[EntityKey]:
        return self.expected_keys & self.detected_keys

    @property
    def fp(self) -> set[EntityKey]:
        return self.detected_keys - self.expected_keys

    @property
    def fn(self) -> set[EntityKey]:
        return self.expected_keys - self.detected_keys

    @property
    def precision(self) -> float:
        detected_count = len(self.detected_keys)
        return len(self.tp) / detected_count if detected_count else 1.0

    @property
    def recall(self) -> float:
        expected_count = len(self.expected_keys)
        return len(self.tp) / expected_count if expected_count else 1.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


@dataclass
class BenchmarkRunResult:
    split: BenchmarkSplit
    rows: list[RowResult]
    overall: dict[str, float]
    by_type: dict[str, dict[str, float]]
    exit_code: int
    threshold_f1: float
    config: AnonLMConfig
    artifact_path: Path | None = None


def load_dataset(csv_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required_columns = {"doc_id", "chunk_id", "input", "expected"}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV has no header row.")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing required columns: {missing_str}")

        for row in reader:
            expected = str(row.get("expected", "")).strip()
            row["expected"] = json.loads(expected) if expected else []
            rows.append(row)

    if not rows:
        raise ValueError("Dataset has no rows.")
    return rows


def entity_to_key(pii_type: str, text: str) -> EntityKey:
    try:
        parsed_type = PIIType(pii_type)
    except ValueError:
        return EntityKey(pii_type=pii_type, canonical=text.strip())

    return EntityKey(pii_type=pii_type, canonical=normalize(text, parsed_type))


def evaluate_row(row: dict[str, Any], engine: AnonymizationEngine) -> RowResult:
    expected_keys = {
        entity_to_key(str(entity["type"]), str(entity["text"]))
        for entity in row["expected"]
    }

    detected_raw = engine.detect_entities(str(row["input"]))
    detected_keys = {
        entity_to_key(str(entity["type"]), str(entity["text"]))
        for entity in detected_raw
    }

    return RowResult(
        doc_id=str(row["doc_id"]),
        chunk_id=str(row["chunk_id"]),
        input=str(row["input"]),
        expected_keys=expected_keys,
        detected_keys=detected_keys,
        detected_raw=detected_raw,
    )


def aggregate_metrics(results: list[RowResult]) -> dict[str, float]:
    all_tp = sum(len(result.tp) for result in results)
    all_fp = sum(len(result.fp) for result in results)
    all_fn = sum(len(result.fn) for result in results)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 1.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": float(all_tp),
        "fp": float(all_fp),
        "fn": float(all_fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_by_type(results: list[RowResult]) -> dict[str, dict[str, float]]:
    tp_by_type: dict[str, int] = defaultdict(int)
    fp_by_type: dict[str, int] = defaultdict(int)
    fn_by_type: dict[str, int] = defaultdict(int)

    for row in results:
        for key in row.tp:
            tp_by_type[key.pii_type] += 1
        for key in row.fp:
            fp_by_type[key.pii_type] += 1
        for key in row.fn:
            fn_by_type[key.pii_type] += 1

    all_types = sorted(set(tp_by_type) | set(fp_by_type) | set(fn_by_type))
    output: dict[str, dict[str, float]] = {}

    for pii_type in all_types:
        tp = tp_by_type[pii_type]
        fp = fp_by_type[pii_type]
        fn = fn_by_type[pii_type]
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        output[pii_type] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return output


def run_benchmark(
    *,
    engine: AnonymizationEngine,
    dataset_path: str,
    split: BenchmarkSplit,
    threshold_f1: float = 0.80,
    verbose: bool = False,
    history_dir: str = "runs/benchmarks",
    save_history: bool = True,
    stream: TextIO | None = None,
) -> BenchmarkRunResult:
    output = stream
    rows = load_dataset(dataset_path)
    selected_rows, total_docs, selected_docs = select_rows_for_split(rows, split)

    if output is not None:
        print(f"\nLoaded {len(rows)} rows ({total_docs} docs) from {dataset_path}", file=output)
        print(
            f"Using split '{split.value}': {len(selected_rows)} rows from {selected_docs} docs",
            file=output,
        )
        print("Running LLM detection...\n", file=output)

    results: list[RowResult] = []
    total = len(selected_rows)
    for index, row in enumerate(selected_rows, 1):
        if output is not None:
            output.write(f"\r  [{index:2d}/{total}] {row['doc_id']} chunk {row['chunk_id']}...")
            output.flush()
        results.append(evaluate_row(row, engine))

    if output is not None:
        output.write("\r" + " " * 60 + "\r")

    overall = aggregate_metrics(results)
    by_type = aggregate_by_type(results)

    if output is not None:
        print_report(results=results, overall=overall, by_type=by_type, verbose=verbose, stream=output)

    exit_code = 0 if overall["f1"] >= threshold_f1 else 1
    benchmark_result = BenchmarkRunResult(
        split=split,
        rows=results,
        overall=overall,
        by_type=by_type,
        exit_code=exit_code,
        threshold_f1=threshold_f1,
        config=engine.config,
    )

    if save_history:
        benchmark_result.artifact_path = write_run_artifacts(
            result=benchmark_result,
            script_name=Path(sys.argv[0]).name,
            dataset_path=dataset_path,
            history_dir=history_dir,
            prompt_text=SYSTEM_PROMPT,
        )

    return benchmark_result
