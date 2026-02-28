# Benchmarking Guide

## Goal

Measure PII detection quality (precision/recall/F1) against a gold dataset, while preserving reproducibility.

## Splits

Splits are document-based and deterministic by `doc_id` appearance order:
- `dev`: first 50% of documents
- `val`: next 35% of documents
- `final`: remaining 15% of documents

## Command

```bash
anonlm benchmark run --dataset datasets/pii_mvp_dataset.csv --split dev --verbose
```

## Recommended protocol

1. Iterate prompts or settings only on `dev`.
2. Compare shortlisted candidates on `val`.
3. Run `final` once for end-state reporting.

## Output and exit codes

- Exit code `0` when overall F1 is above or equal to threshold (`--threshold-f1`, default `0.80`).
- Exit code `1` otherwise.

## Reproducibility metadata

Each persisted run stores:
- timestamp and split
- model/provider runtime settings (excluding secret values)
- dataset path and SHA256 hash
- prompt SHA256 and prompt text
- git commit hash (if available)
- overall and per-type metrics
- per-row expected/detected/TP/FP/FN detail
