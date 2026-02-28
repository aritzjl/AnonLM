"""State type used by the LangGraph pipeline."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class PIIState(TypedDict):
    original_text: str
    chunks: list[str]
    chunk_index: int
    all_entities: Annotated[list[dict[str, str]], operator.add]
    mapping_forward: dict[str, str]
    mapping_reverse: dict[str, str]
    type_counters: dict[str, int]
    anonymized_text: str
