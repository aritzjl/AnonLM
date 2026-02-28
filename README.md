# AnonLM

AnonLM is an open-source Python library for LLM-based PII anonymization with reproducible benchmarking.

It provides:
- A configurable anonymization engine for OpenAI-compatible providers.
- A stable Python API for anonymize/deanonymize workflows.
- A unified CLI for anonymization and benchmark execution.
- Benchmark history artifacts for auditability and experiment tracking.

## Installation

```bash
pip install anonlm
```

For development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,test]
```

## Quickstart (Python API)

```python
from anonlm import anonymize

result = anonymize("Contact Jane Doe at jane.doe@example.com or +34 600 123 456.")
print(result.anonymized_text)
print(result.mapping_forward)
```

## Quickstart (CLI)

```bash
# Text input
anonlm anonymize --text "Contact Jane Doe at jane.doe@example.com"

# File input -> JSON output
anonlm anonymize --file input.txt --output output.json

# Benchmark run
anonlm benchmark run --dataset datasets/pii_mvp_dataset.csv --split dev
```

## Configuration

Configuration precedence is:
1. Explicit CLI flags
2. Environment variables (`ANONLM_*`)
3. Provider defaults

Core environment variables:

| Variable | Description |
| --- | --- |
| `ANONLM_PROVIDER` | `openai`, `openrouter`, `groq`, or `custom` |
| `ANONLM_MODEL_NAME` | Model identifier |
| `ANONLM_BASE_URL` | OpenAI-compatible base URL |
| `ANONLM_API_KEY_ENV` | Env var name containing API key |
| `ANONLM_API_KEY` | API key value |
| `ANONLM_TEMPERATURE` | LLM temperature |
| `ANONLM_MAX_CHUNK_CHARS` | Chunk size |
| `ANONLM_CHUNK_OVERLAP_CHARS` | Chunk overlap |

Provider examples:

```bash
# OpenAI
export ANONLM_PROVIDER=openai
export ANONLM_API_KEY=sk-...

# OpenRouter
export ANONLM_PROVIDER=openrouter
export ANONLM_API_KEY=...
export ANONLM_MODEL_NAME=openai/gpt-4o-mini

# Groq
export ANONLM_PROVIDER=groq
export ANONLM_API_KEY=...
export ANONLM_MODEL_NAME=llama-3.3-70b-versatile

# Custom OpenAI-compatible endpoint
export ANONLM_PROVIDER=custom
export ANONLM_BASE_URL=https://your.endpoint/v1
export ANONLM_API_KEY=...
```

## Benchmarking

Run benchmark with deterministic document-based splits (`dev`, `val`, `final`):

```bash
anonlm benchmark run --dataset datasets/pii_mvp_dataset.csv --split dev --verbose
```

Optional benchmark controls:

```bash
anonlm benchmark run \
  --dataset datasets/pii_mvp_dataset.csv \
  --split val \
  --history-dir runs/benchmarks \
  --threshold-f1 0.80
```

Artifacts:
- JSON run detail: `runs/benchmarks/<timestamp>__<split>.json`
- CSV summary index: `runs/benchmarks/index.csv`

See `docs/benchmarking.md` for protocol and interpretation guidelines.

## Public API

- `anonlm.anonymize(text: str, config: AnonLMConfig | None = None) -> AnonymizationResult`
- `anonlm.deanonymize(text: str, mapping_reverse: dict[str, str]) -> str`
- `anonlm.create_engine(config: AnonLMConfig | None = None) -> AnonymizationEngine`

## Project status

Current status: `0.x` (early API hardening). Expect minor breaking changes until `1.0.0`.

## License

Apache-2.0
