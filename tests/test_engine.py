from __future__ import annotations

from dataclasses import dataclass

from anonlm.config import AnonLMConfig
from anonlm.engine import AnonymizationEngine
from anonlm.schema import PIIEntity, PIIResponse, PIIType


@dataclass
class SequenceLLM:
    responses: list[PIIResponse]

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


def test_engine_detect_entities() -> None:
    llm = SequenceLLM(
        responses=[PIIResponse(entities=[PIIEntity(type=PIIType.EMAIL, text="user@example.com")])]
    )
    engine = AnonymizationEngine(config=AnonLMConfig(api_key="test-key"), llm=llm)

    entities = engine.detect_entities("user@example.com")
    assert len(entities) == 1
    assert entities[0]["type"] == "EMAIL"
