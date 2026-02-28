# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [0.1.5] - 2026-02-28
### Added
- Linking metadata now includes detailed merge traces per entity link: source/target text, canonical values, and token mapping (`from_token` → `to_token`).

### Changed
- Updated node and engine tests, plus README docs, to cover enriched linking details.

## [0.1.4] - 2026-02-28
### Added
- `AnonymizationResult` now exposes linking metadata in `result.linking`.
- JSON serialization now includes `linking` (`link_count`, `links`) in `result.to_dict()` and CLI output.

### Changed
- Updated engine/CLI tests and README examples to cover linking metadata.

## [0.1.3] - 2026-02-28
### Changed
- Applied Ruff-driven cleanup and formatting updates across CLI and benchmarking modules.
- Updated tests to satisfy lint constraints without behavior changes.

## [0.1.2] - 2026-02-28
### Added
- Cross-chunk person entity linking stage to merge aliases that refer to the same person in a full document.
- New linking prompt and schema models for structured alias linking output.
- Test coverage for linking behavior in graph/node and end-to-end engine flow.

### Changed
- Graph flow now routes processed chunks through a linking node before final anonymization.
- Token counters and reverse mappings are recomputed after alias merges to keep mappings consistent.

## [0.1.1] - 2026-02-28
### Added
- `AnonymizationResult` now includes chunking metadata (`chunk_count`, `chunks`, `max_chunk_chars`, `chunk_overlap_chars`).
- CLI JSON output now includes `chunking` in anonymization payloads.

### Changed
- Updated tests and README examples to cover chunking metadata in responses.

## [0.1.0] - 2026-02-28
### Added
- Initial standalone `anonlm` package.
- PII anonymization engine based on LangGraph and OpenAI-compatible APIs.
- Unified CLI for anonymization and benchmark execution.
- Reproducible benchmark artifacts and dataset split handling.
- OSS scaffolding for CI, release, contribution, and security workflows.
