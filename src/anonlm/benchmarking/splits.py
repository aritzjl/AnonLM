"""Benchmark split logic."""

from __future__ import annotations

import math
from enum import Enum


class BenchmarkSplit(str, Enum):
    DEV = "dev"
    VAL = "val"
    FINAL = "final"


def get_ordered_doc_ids(rows: list[dict[str, object]]) -> list[str]:
    ordered_doc_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered_doc_ids.append(doc_id)
    return ordered_doc_ids


def build_doc_splits(ordered_doc_ids: list[str]) -> dict[BenchmarkSplit, set[str]]:
    total_docs = len(ordered_doc_ids)
    first_cut = math.ceil(total_docs * 0.50)
    second_docs = round(total_docs * 0.35)
    second_cut = min(total_docs, first_cut + second_docs)

    return {
        BenchmarkSplit.DEV: set(ordered_doc_ids[:first_cut]),
        BenchmarkSplit.VAL: set(ordered_doc_ids[first_cut:second_cut]),
        BenchmarkSplit.FINAL: set(ordered_doc_ids[second_cut:]),
    }


def select_rows_for_split(
    rows: list[dict[str, object]], split: BenchmarkSplit
) -> tuple[list[dict[str, object]], int, int]:
    ordered_doc_ids = get_ordered_doc_ids(rows)
    split_map = build_doc_splits(ordered_doc_ids)
    selected_doc_ids = split_map[split]
    selected_rows = [row for row in rows if str(row["doc_id"]) in selected_doc_ids]
    return selected_rows, len(ordered_doc_ids), len(selected_doc_ids)
