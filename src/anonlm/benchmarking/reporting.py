"""Benchmark report rendering utilities."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:  # pragma: no cover
    from anonlm.benchmarking.runner import RowResult

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m~\033[0m"


def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_row_detail(result: "RowResult", stream: TextIO) -> None:
    icon = PASS if not result.fn and not result.fp else (WARN if result.tp else FAIL)
    print(
        f"\n  {icon} {result.doc_id} chunk {result.chunk_id}  "
        f"P={result.precision:.2f} R={result.recall:.2f} F1={result.f1:.2f}",
        file=stream,
    )
    preview = result.input[:80]
    suffix = "…" if len(result.input) > 80 else ""
    print(f"     Input : {preview}{suffix}", file=stream)

    for key in sorted(result.tp, key=lambda x: x.pii_type):
        print(f"     \033[32m  TP\033[0m  [{key.pii_type:10s}] {key.canonical!r}", file=stream)
    for key in sorted(result.fp, key=lambda x: x.pii_type):
        print(f"     \033[31m  FP\033[0m  [{key.pii_type:10s}] {key.canonical!r}", file=stream)
    for key in sorted(result.fn, key=lambda x: x.pii_type):
        print(f"     \033[33m  FN\033[0m  [{key.pii_type:10s}] {key.canonical!r}", file=stream)


def print_report(
    *,
    results: list["RowResult"],
    overall: dict[str, float],
    by_type: dict[str, dict[str, float]],
    verbose: bool,
    stream: TextIO | None = None,
) -> None:
    output = stream or sys.stdout

    if verbose:
        print("\n" + "═" * 70, file=output)
        print("  PER-ROW DETAIL", file=output)
        print("═" * 70, file=output)
        for result in results:
            print_row_detail(result, output)

    print("\n" + "═" * 70, file=output)
    print("  PER-TYPE METRICS", file=output)
    print("═" * 70, file=output)
    print(f"  {'Type':<12} {'TP':>4} {'FP':>4} {'FN':>4}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}", file=output)
    print("  " + "-" * 62, file=output)

    for pii_type, metrics in by_type.items():
        print(
            f"  {pii_type:<12} {int(metrics['tp']):>4} {int(metrics['fp']):>4} {int(metrics['fn']):>4}"
            f"  {metrics['precision']:>8.1%}  {metrics['recall']:>7.1%}  {metrics['f1']:>5.1%}"
            f"  {bar(metrics['f1'], 12)}",
            file=output,
        )

    print("\n" + "═" * 70, file=output)
    print("  OVERALL METRICS", file=output)
    print("═" * 70, file=output)
    print(f"  Rows evaluated : {len(results)}", file=output)
    print(f"  True Positives : {int(overall['tp'])}", file=output)
    print(f"  False Positives: {int(overall['fp'])}", file=output)
    print(f"  False Negatives: {int(overall['fn'])}", file=output)
    print(f"  Precision      : {overall['precision']:.1%}  {bar(overall['precision'])}", file=output)
    print(f"  Recall         : {overall['recall']:.1%}  {bar(overall['recall'])}", file=output)
    print(f"  F1             : {overall['f1']:.1%}  {bar(overall['f1'])}", file=output)

    negative_rows = [row for row in results if not row.expected_keys]
    negative_fp = sum(len(row.fp) for row in negative_rows)
    print(f"\n  Negative rows  : {len(negative_rows)}  (expected no PII)", file=output)
    print(f"  FP on negatives: {negative_fp}", file=output)
    print("═" * 70, file=output)
