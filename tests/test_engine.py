from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anonlm.config import AnonLMConfig
from anonlm.engine import AnonymizationEngine
from anonlm.schema import PIIEntity, PIILink, PIILinkingResponse, PIIResponse, PIIType


@dataclass
class SequenceLLM:
    responses: list[Any]

    def __post_init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_engine_anonymize_tokens() -> None:
    llm = SequenceLLM(
        responses=[
            PIIResponse(
                entities=[
                    PIIEntity(type=PIIType.PERSON, text="Maria Garcia"),
                    PIIEntity(type=PIIType.ORG, text="Acme Corp"),
                ]
            )
        ]
    )
    engine = AnonymizationEngine(config=AnonLMConfig(api_key="test-key"), llm=llm)

    result = engine.anonymize("Maria Garcia works at Acme Corp.")
    assert "[[PERSON_1]]" in result.anonymized_text
    assert "[[ORG_1]]" in result.anonymized_text
    assert result.mapping_forward["Maria Garcia"] == "[[PERSON_1]]"
    assert result.chunking.chunk_count == 1
    assert result.chunking.chunks == ["Maria Garcia works at Acme Corp."]


def test_engine_same_entity_same_token_across_chunks() -> None:
    llm = SequenceLLM(
        responses=[
            PIIResponse(entities=[PIIEntity(type=PIIType.PERSON, text="Maria Garcia")]),
            PIIResponse(entities=[PIIEntity(type=PIIType.PERSON, text="Maria Garcia")]),
        ]
    )
    config = AnonLMConfig(api_key="test-key", max_chunk_chars=32, chunk_overlap_chars=8)
    engine = AnonymizationEngine(config=config, llm=llm)

    text = "Maria Garcia works here.\n\nMaria Garcia comes back tomorrow."
    result = engine.anonymize(text)

    assert llm.calls == 2
    assert list(result.mapping_forward.values()).count("[[PERSON_1]]") == 1
    assert "[[PERSON_2]]" not in result.mapping_forward.values()
    assert result.chunking.chunk_count == 2


def test_engine_to_dict_contains_chunking_metadata() -> None:
    llm = SequenceLLM(
        responses=[PIIResponse(entities=[PIIEntity(type=PIIType.PERSON, text="Jane Doe")])]
    )
    config = AnonLMConfig(api_key="test-key", max_chunk_chars=50, chunk_overlap_chars=10)
    engine = AnonymizationEngine(config=config, llm=llm)

    result = engine.anonymize("Jane Doe is here.")
    payload = result.to_dict()

    assert "chunking" in payload
    assert payload["chunking"]["chunk_count"] == 1
    assert payload["chunking"]["chunks"] == ["Jane Doe is here."]
    assert payload["chunking"]["max_chunk_chars"] == 50
    assert payload["chunking"]["chunk_overlap_chars"] == 10
    assert payload["linking"]["link_count"] == 0
    assert payload["linking"]["links"] == []


def test_engine_detect_entities() -> None:
    llm = SequenceLLM(
        responses=[PIIResponse(entities=[PIIEntity(type=PIIType.EMAIL, text="user@example.com")])]
    )
    engine = AnonymizationEngine(config=AnonLMConfig(api_key="test-key"), llm=llm)

    entities = engine.detect_entities("user@example.com")
    assert len(entities) == 1
    assert entities[0]["type"] == "EMAIL"


def test_engine_links_person_aliases_across_chunks() -> None:
    llm = SequenceLLM(
        responses=[
            PIIResponse(entities=[PIIEntity(type=PIIType.PERSON, text="Sarah Johnson")]),
            PIIResponse(entities=[PIIEntity(type=PIIType.PERSON, text="Sarah")]),
            PIILinkingResponse(
                links=[
                    PIILink(
                        type=PIIType.PERSON,
                        representative="Sarah Johnson",
                        aliases=["Sarah"],
                    )
                ]
            ),
        ]
    )
    config = AnonLMConfig(api_key="test-key", max_chunk_chars=32, chunk_overlap_chars=8)
    engine = AnonymizationEngine(config=config, llm=llm)

    text = "My name is Sarah Johnson.\n\nBest regards,\nSarah"
    result = engine.anonymize(text)

    assert llm.calls == 3
    assert result.mapping_forward["Sarah Johnson"] == "[[PERSON_1]]"
    assert result.mapping_forward["Sarah"] == "[[PERSON_1]]"
    assert "[[PERSON_2]]" not in result.mapping_reverse
    assert result.anonymized_text.count("[[PERSON_1]]") >= 2
    assert result.linking.link_count == 1
    assert result.linking.links[0] == {
        "type": "PERSON",
        "from": "Sarah",
        "to": "Sarah Johnson",
    }
