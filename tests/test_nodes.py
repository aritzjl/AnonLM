from __future__ import annotations

from dataclasses import dataclass

from anonlm.config import AnonLMConfig
from anonlm.nodes import _chunk_text, make_anonymize_node, make_prepare_node, make_process_chunk_node
from anonlm.schema import PIIEntity, PIIResponse, PIIType


@dataclass
class FakeLLM:
    response: PIIResponse

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
