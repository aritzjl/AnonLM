"""Public engine implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None

from anonlm.config import AnonLMConfig
from anonlm.graph import create_graph


@dataclass(frozen=True)
class ChunkingMetadata:
    chunk_count: int
    chunks: list[str]
    max_chunk_chars: int
    chunk_overlap_chars: int

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "chunks": self.chunks,
            "max_chunk_chars": self.max_chunk_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
        }


@dataclass(frozen=True)
class LinkingMetadata:
    link_count: int
    links: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "link_count": self.link_count,
            "links": self.links,
        }


@dataclass(frozen=True)
class AnonymizationResult:
    anonymized_text: str
    mapping_forward: dict[str, str]
    mapping_reverse: dict[str, str]
    all_entities: list[dict[str, str]]
    type_counters: dict[str, int]
    chunking: ChunkingMetadata = field(
        default_factory=lambda: ChunkingMetadata(
            chunk_count=0,
            chunks=[],
            max_chunk_chars=0,
            chunk_overlap_chars=0,
        )
    )
    linking: LinkingMetadata = field(
        default_factory=lambda: LinkingMetadata(
            link_count=0,
            links=[],
        )
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "anonymized_text": self.anonymized_text,
            "mapping_forward": self.mapping_forward,
            "mapping_reverse": self.mapping_reverse,
            "all_entities": self.all_entities,
            "type_counters": self.type_counters,
            "chunking": self.chunking.to_dict(),
            "linking": self.linking.to_dict(),
        }


class AnonymizationEngine:
    def __init__(self, config: AnonLMConfig | None = None, llm: Any | None = None):
        self.config = (config or AnonLMConfig.from_env()).validate()
        self._llm = llm if llm is not None else _build_llm(self.config)
        self._app = create_graph(self.config, self._llm)

    def anonymize(self, text: str) -> AnonymizationResult:
        source = text.strip()
        if not source:
            raise ValueError("Text must not be empty.")

        result = self._app.invoke({"original_text": source})
        entity_links = result.get("entity_links", [])
        return AnonymizationResult(
            anonymized_text=result["anonymized_text"],
            mapping_forward=result["mapping_forward"],
            mapping_reverse=result["mapping_reverse"],
            all_entities=result["all_entities"],
            type_counters=result["type_counters"],
            chunking=ChunkingMetadata(
                chunk_count=len(result["chunks"]),
                chunks=result["chunks"],
                max_chunk_chars=self.config.max_chunk_chars,
                chunk_overlap_chars=self.config.chunk_overlap_chars,
            ),
            linking=LinkingMetadata(
                link_count=len(entity_links),
                links=entity_links,
            ),
        )

    def detect_entities(self, text: str) -> list[dict[str, str]]:
        return self.anonymize(text).all_entities


def _build_llm(config: AnonLMConfig) -> Any:
    if ChatOpenAI is None:
        return None

    return ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        api_key=config.resolved_api_key(),
        base_url=config.base_url,
    )
