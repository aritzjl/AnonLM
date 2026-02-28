"""Utilities for deanonymization."""

from __future__ import annotations


def deanonymize_text(text: str, mapping_reverse: dict[str, str]) -> str:
    out = text
    for token in sorted(mapping_reverse, key=len, reverse=True):
        out = out.replace(token, mapping_reverse[token])
    return out
