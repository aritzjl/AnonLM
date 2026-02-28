"""Persistence utilities for benchmark runs."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from anonlm.benchmarking.runner import BenchmarkRunResult, RowResult


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or "unknown"
    except Exception:
        return "unknown"


def _build_row_payload(row: RowResult) -> dict[str, object]:
    return {
        "doc_id": row.doc_id,
        "chunk_id": row.chunk_id,
        "input": row.input,
        "expected": sorted(
            [{"type": key.pii_type, "canonical": key.canonical} for key in row.expected_keys],
            key=lambda item: (item["type"], item["canonical"]),
        ),
        "detected": sorted(
            [{"type": key.pii_type, "canonical": key.canonical} for key in row.detected_keys],
            key=lambda item: (item["type"], item["canonical"]),
        ),
        "tp": sorted(
            [{"type": key.pii_type, "canonical": key.canonical} for key in row.tp],
            key=lambda item: (item["type"], item["canonical"]),
        ),
        "fp": sorted(
            [{"type": key.pii_type, "canonical": key.canonical} for key in row.fp],
            key=lambda item: (item["type"], item["canonical"]),
        ),
        "fn": sorted(
            [{"type": key.pii_type, "canonical": key.canonical} for key in row.fn],
            key=lambda item: (item["type"], item["canonical"]),
        ),
        "metrics": {
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
        },
        "detected_raw": row.detected_raw,
    }


def write_run_artifacts(
    *,
    result: BenchmarkRunResult,
    script_name: str,
    dataset_path: str,
    history_dir: str,
    prompt_text: str,
) -> Path:
    now = dt.datetime.now(dt.timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")

    history_root = Path(history_dir)
    history_root.mkdir(parents=True, exist_ok=True)

    dataset_abs = str(Path(dataset_path).resolve())
    dataset_hash = _sha256_file(dataset_path)
    prompt_hash = _sha256_text(prompt_text)

    payload = {
        "run_id": f"{timestamp}_{result.split.value}",
        "timestamp_utc": now.isoformat(),
        "script": script_name,
        "split": result.split.value,
        "exit_code": result.exit_code,
        "threshold_f1": result.threshold_f1,
        "dataset": {
            "path": dataset_abs,
            "sha256": dataset_hash,
        },
        "git": {
            "commit": _safe_git_commit(),
        },
        "config": result.config.to_public_dict(),
        "prompt": {
            "sha256": prompt_hash,
            "text": prompt_text,
        },
        "metrics": {
            "overall": result.overall,
            "by_type": result.by_type,
        },
        "rows": [_build_row_payload(row) for row in result.rows],
    }

    artifact_path = history_root / f"{timestamp}__{result.split.value}.json"
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    index_path = history_root / "index.csv"
    write_header = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "timestamp_utc",
                    "run_id",
                    "script",
                    "split",
                    "dataset_path",
                    "dataset_sha256",
                    "git_commit",
                    "provider",
                    "model_name",
                    "base_url",
                    "api_key_env",
                    "temperature",
                    "max_chunk_chars",
                    "chunk_overlap_chars",
                    "prompt_sha256",
                    "rows_evaluated",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "exit_code",
                    "artifact_json",
                ]
            )

        config = result.config
        writer.writerow(
            [
                payload["timestamp_utc"],
                payload["run_id"],
                script_name,
                result.split.value,
                dataset_abs,
                dataset_hash,
                payload["git"]["commit"],
                config.provider.value,
                config.model_name,
                config.base_url,
                config.api_key_env,
                config.temperature,
                config.max_chunk_chars,
                config.chunk_overlap_chars,
                prompt_hash,
                len(result.rows),
                int(result.overall["tp"]),
                int(result.overall["fp"]),
                int(result.overall["fn"]),
                result.overall["precision"],
                result.overall["recall"],
                result.overall["f1"],
                result.exit_code,
                str(artifact_path.resolve()),
            ]
        )

    return artifact_path
