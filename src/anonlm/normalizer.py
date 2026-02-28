"""Deterministic normalization for matching PII entities."""

from __future__ import annotations

import re

from anonlm.schema import PIIType


def normalize(text: str, pii_type: PIIType) -> str:
    if pii_type == PIIType.EMAIL:
        return text.lower().strip()

    if pii_type == PIIType.PHONE:
        digits = re.sub(r"[^\d]", "", text)
        return "+" + digits if text.strip().startswith("+") else digits

    if pii_type == PIIType.ID_NUMBER:
        return re.sub(r"[\s\-.]", "", text).upper()

    if pii_type in (PIIType.PERSON, PIIType.ORG):
        return re.sub(r"\s+", " ", text).strip()

    return text.strip()
