"""Stable public API wrappers."""

from __future__ import annotations

from anonlm.config import AnonLMConfig
from anonlm.deanonymize import deanonymize_text
from anonlm.engine import AnonymizationEngine, AnonymizationResult


def create_engine(config: AnonLMConfig | None = None) -> AnonymizationEngine:
    return AnonymizationEngine(config=config)


def anonymize(text: str, config: AnonLMConfig | None = None) -> AnonymizationResult:
    engine = create_engine(config)
    return engine.anonymize(text)


def deanonymize(text: str, mapping_reverse: dict[str, str]) -> str:
    return deanonymize_text(text, mapping_reverse)
