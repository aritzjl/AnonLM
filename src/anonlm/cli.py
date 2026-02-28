"""Command-line interface for AnonLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anonlm.api import create_engine
from anonlm.benchmarking import BenchmarkSplit, run_benchmark
from anonlm.config import AnonLMConfig, ProviderPreset


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=[value.value for value in ProviderPreset])
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-chunk-chars", type=int)
    parser.add_argument("--chunk-overlap-chars", type=int)


def _build_config_from_args(args: argparse.Namespace) -> AnonLMConfig:
    base_config = AnonLMConfig.from_env()
    overrides: dict[str, object] = {
        "provider": args.provider,
        "model_name": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "max_chunk_chars": args.max_chunk_chars,
        "chunk_overlap_chars": args.chunk_overlap_chars,
    }
    return base_config.with_overrides(**overrides)


def _command_anonymize(args: argparse.Namespace) -> int:
    config = _build_config_from_args(args)
    engine = create_engine(config)

    if args.text is not None:
        source_text = args.text
    else:
        source_text = Path(args.file).read_text(encoding="utf-8")

    result = engine.anonymize(source_text)
    payload = result.to_dict()

    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


def _command_benchmark_run(args: argparse.Namespace) -> int:
    config = _build_config_from_args(args)
    engine = create_engine(config)
    benchmark_result = run_benchmark(
        engine=engine,
        dataset_path=args.dataset,
        split=BenchmarkSplit(args.split),
        threshold_f1=args.threshold_f1,
        verbose=args.verbose,
        history_dir=args.history_dir,
        save_history=not args.no_save_history,
        stream=sys.stdout,
    )

    if args.no_save_history:
        print("\nHistory persistence disabled (--no-save-history).")
    elif benchmark_result.artifact_path is not None:
        print(f"\nSaved benchmark artifact: {benchmark_result.artifact_path}")
        print(f"Updated benchmark index: {Path(args.history_dir) / 'index.csv'}")

    return benchmark_result.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anonlm", description="AnonLM CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    anonymize_parser = subparsers.add_parser("anonymize", help="Anonymize input text")
    source_group = anonymize_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="Inline text to anonymize")
    source_group.add_argument("--file", help="Path to UTF-8 text file to anonymize")
    anonymize_parser.add_argument("--output", help="Path to JSON output file")
    _add_common_config_args(anonymize_parser)
    anonymize_parser.set_defaults(func=_command_anonymize)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark commands")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)

    benchmark_run_parser = benchmark_subparsers.add_parser("run", help="Run benchmark")
    benchmark_run_parser.add_argument("--dataset", default="datasets/pii_mvp_dataset.csv")
    benchmark_run_parser.add_argument("--split", choices=[value.value for value in BenchmarkSplit], default="dev")
    benchmark_run_parser.add_argument("--verbose", action="store_true")
    benchmark_run_parser.add_argument("--history-dir", default="runs/benchmarks")
    benchmark_run_parser.add_argument("--no-save-history", action="store_true")
    benchmark_run_parser.add_argument("--threshold-f1", type=float, default=0.80)
    _add_common_config_args(benchmark_run_parser)
    benchmark_run_parser.set_defaults(func=_command_benchmark_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
