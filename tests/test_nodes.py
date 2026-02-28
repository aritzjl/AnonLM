from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anonlm.config import AnonLMConfig
from anonlm.nodes import (
    _chunk_text,
    make_anonymize_node,
    make_link_entities_node,
    make_prepare_node,
    make_process_chunk_node,
)
from anonlm.schema import PIIEntity, PIILink, PIILinkingResponse, PIIResponse, PIIType


@dataclass
class FakeLLM:
    response: Any

    def invoke(self, messages):  # noqa: ANN001
        return self.response


def test_chunk_text_short_paragraph() -> None:
    chunks = _chunk_text("Hello world.", max_chunk_chars=1000, chunk_overlap_chars=100)
    assert chunks == ["Hello world."]


def test_chunk_text_two_paragraphs() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    chunks = _chunk_text(text, max_chunk_chars=1000, chunk_overlap_chars=100)
    assert len(chunks) == 2


def test_chunk_text_long_paragraph_splits() -> None:
    sentence = "This is a sentence that is not too long. "
    long_para = sentence * 50
    chunks = _chunk_text(long_para, max_chunk_chars=200, chunk_overlap_chars=30)
    assert len(chunks) > 1


def test_prepare_node_returns_chunks() -> None:
    config = AnonLMConfig()
    prepare = make_prepare_node(config)
    result = prepare({"original_text": "Hello.\n\nWorld."})
    assert result["chunks"] == ["Hello.", "World."]
    assert result["chunk_index"] == 0


def test_process_chunk_creates_token() -> None:
    llm = FakeLLM(
        response=PIIResponse(entities=[PIIEntity(type=PIIType.PERSON, text="Maria Garcia")])
    )
    process_chunk = make_process_chunk_node(llm)
    state = {
        "chunks": ["Hello Maria Garcia"],
        "chunk_index": 0,
        "all_entities": [],
        "mapping_forward": {},
        "mapping_reverse": {},
        "type_counters": {},
    }

    result = process_chunk(state)
    assert result["chunk_index"] == 1
    assert result["mapping_forward"]["Maria Garcia"] == "[[PERSON_1]]"
    assert result["type_counters"]["PERSON"] == 1


def test_process_chunk_deduplicates_canonical() -> None:
    llm = FakeLLM(
        response=PIIResponse(entities=[PIIEntity(type=PIIType.EMAIL, text="USER@EXAMPLE.COM")])
    )
    process_chunk = make_process_chunk_node(llm)
    state = {
        "chunks": ["USER@EXAMPLE.COM"],
        "chunk_index": 0,
        "all_entities": [],
        "mapping_forward": {"user@example.com": "[[EMAIL_1]]"},
        "mapping_reverse": {"[[EMAIL_1]]": "user@example.com"},
        "type_counters": {"EMAIL": 1},
    }

    result = process_chunk(state)
    assert result["type_counters"]["EMAIL"] == 1
    assert len(result["mapping_forward"]) == 1


def test_anonymize_node_replaces_text() -> None:
    anonymize_node = make_anonymize_node()
    state = {
        "original_text": "Call Maria Garcia at +34612345678.",
        "mapping_forward": {"Maria Garcia": "[[PERSON_1]]", "+34612345678": "[[PHONE_1]]"},
        "mapping_reverse": {"[[PERSON_1]]": "Maria Garcia", "[[PHONE_1]]": "+34 612 345 678"},
    }

    result = anonymize_node(state)
    assert "[[PERSON_1]]" in result["anonymized_text"]
    assert "[[PHONE_1]]" in result["anonymized_text"]


def test_anonymize_node_longer_match_first() -> None:
    anonymize_node = make_anonymize_node()
    state = {
        "original_text": "John Smith and John are here.",
        "mapping_forward": {"John Smith": "[[PERSON_1]]", "John": "[[PERSON_2]]"},
        "mapping_reverse": {"[[PERSON_1]]": "John Smith", "[[PERSON_2]]": "John"},
    }

    result = anonymize_node(state)
    assert result["anonymized_text"] == "[[PERSON_1]] and [[PERSON_2]] are here."


def test_link_entities_node_merges_person_aliases() -> None:
    llm = FakeLLM(
        response=PIILinkingResponse(
            links=[
                PIILink(type=PIIType.PERSON, representative="Sarah Johnson", aliases=["Sarah"])
            ]
        )
    )
    link_entities = make_link_entities_node(llm)
    state = {
        "original_text": "My name is Sarah Johnson. Best regards, Sarah.",
        "all_entities": [
            {
                "type": "PERSON",
                "text": "Sarah Johnson",
                "canonical": "Sarah Johnson",
                "token": "[[PERSON_1]]",
            },
            {"type": "PERSON", "text": "Sarah", "canonical": "Sarah", "token": "[[PERSON_2]]"},
        ],
        "mapping_forward": {"Sarah Johnson": "[[PERSON_1]]", "Sarah": "[[PERSON_2]]"},
        "mapping_reverse": {"[[PERSON_1]]": "Sarah Johnson", "[[PERSON_2]]": "Sarah"},
        "type_counters": {"PERSON": 2},
    }

    result = link_entities(state)
    assert result["mapping_forward"]["Sarah"] == "[[PERSON_1]]"
    assert "[[PERSON_2]]" not in result["mapping_reverse"]
    assert result["type_counters"]["PERSON"] == 1
    assert result["mapping_reverse"]["[[PERSON_1]]"] == "Sarah Johnson"
    assert result["entity_links"] == [
        {
            "type": "PERSON",
            "from": "Sarah",
            "to": "Sarah Johnson",
            "from_canonical": "Sarah",
            "to_canonical": "Sarah Johnson",
            "from_token": "[[PERSON_2]]",
            "to_token": "[[PERSON_1]]",
        }
    ]
