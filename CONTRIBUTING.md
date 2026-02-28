# Contributing to AnonLM

Thanks for contributing.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,test]
```

## Local checks

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Commit and PR expectations

- Keep PRs focused and small.
- Add or update tests for behavior changes.
- Update docs and changelog when relevant.
- Ensure CI is green before requesting review.

## Benchmark changes

If your change affects detection behavior, include benchmark comparison notes:
- split used (`dev`, `val`, `final`)
- overall metrics (P/R/F1)
- per-type deltas and notable FP/FN changes
