"""LangGraph nodes for preparation, extraction and anonymization."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from anonlm.config import AnonLMConfig
from anonlm.normalizer import normalize
from anonlm.prompts import LINKING_PROMPT, SYSTEM_PROMPT
from anonlm.schema import PIILinkingResponse, PIIResponse
from anonlm.state import PIIState

_TOKEN_TYPE_RE = re.compile(r"^\[\[([A-Z_]+)_[0-9]+\]\]$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out).strip()
    return str(content).strip()


def _parse_model_from_message(raw_message: Any, model: type[_ModelT]) -> _ModelT:
    raw_text = _extract_text_content(getattr(raw_message, "content", raw_message))
    if not raw_text:
        raise OutputParserException("Empty LLM response while expecting JSON.")

    try:
        return model.model_validate_json(raw_text)
    except ValidationError:
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if match:
            try:
                return model.model_validate_json(match.group(0))
            except ValidationError:
                pass
        raise OutputParserException(
            f"Failed to parse structured PII JSON. Raw response: {raw_text[:240]!r}"
        ) from None


def _parse_pii_response_from_message(raw_message: Any) -> PIIResponse:
    return _parse_model_from_message(raw_message, PIIResponse)


def _parse_linking_response_from_message(raw_message: Any) -> PIILinkingResponse:
    return _parse_model_from_message(raw_message, PIILinkingResponse)


def _chunk_text(text: str, max_chunk_chars: int, chunk_overlap_chars: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= max_chunk_chars:
            if para.strip():
                chunks.append(para)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        window: list[str] = []
        window_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if window_len + sentence_len > max_chunk_chars and window:
                chunk = " ".join(window)
                if chunk.strip():
                    chunks.append(chunk)

                tail: list[str] = []
                tail_len = 0
                for existing in reversed(window):
                    if tail_len + len(existing) <= chunk_overlap_chars:
                        tail.insert(0, existing)
                        tail_len += len(existing) + 1
                    else:
                        break
                window = tail
                window_len = tail_len

            window.append(sentence)
            window_len += sentence_len + 1

        if window:
            chunk = " ".join(window)
            if chunk.strip():
                chunks.append(chunk)

    return chunks if chunks else [text]


def make_prepare_node(config: AnonLMConfig) -> Callable[[PIIState], dict[str, Any]]:
    def prepare_node(state: PIIState) -> dict[str, Any]:
        chunks = _chunk_text(
            state["original_text"],
            max_chunk_chars=config.max_chunk_chars,
            chunk_overlap_chars=config.chunk_overlap_chars,
        )
        return {
            "chunks": chunks,
            "chunk_index": 0,
            "mapping_forward": {},
            "mapping_reverse": {},
            "type_counters": {},
        }

    return prepare_node


def make_process_chunk_node(
    llm: Any,
) -> Callable[[PIIState], dict[str, Any]]:
    def process_chunk_node(state: PIIState) -> dict[str, Any]:
        if llm is None:
            raise RuntimeError(
                "Missing dependency or model client. Install dependencies and configure API key."
            )

        idx = state["chunk_index"]
        chunk = state["chunks"][idx]

        raw_response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=chunk)]
        )
        response = (
            raw_response
            if isinstance(raw_response, PIIResponse)
            else _parse_pii_response_from_message(raw_response)
        )

        mapping_forward = dict(state["mapping_forward"])
        mapping_reverse = dict(state["mapping_reverse"])
        type_counters = dict(state["type_counters"])
        new_entities: list[dict[str, str]] = []

        for entity in response.entities:
            canonical = normalize(entity.text, entity.type)
            pii_type_str = entity.type.value

            if canonical not in mapping_forward:
                count = type_counters.get(pii_type_str, 0) + 1
                type_counters[pii_type_str] = count
                token = f"[[{pii_type_str}_{count}]]"
                mapping_forward[canonical] = token
                mapping_reverse[token] = entity.text

            new_entities.append(
                {
                    "type": pii_type_str,
                    "text": entity.text,
                    "canonical": canonical,
                    "token": mapping_forward[canonical],
                }
            )

        return {
            "chunk_index": idx + 1,
            "all_entities": new_entities,
            "mapping_forward": mapping_forward,
            "mapping_reverse": mapping_reverse,
            "type_counters": type_counters,
        }

    return process_chunk_node


def _build_linking_payload(state: PIIState) -> str:
    entities_payload: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for entity in state["all_entities"]:
        if entity["type"] != "PERSON":
            continue

        key = (entity["type"], entity["text"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        entities_payload.append({"type": entity["type"], "text": entity["text"]})

    payload = {"text": state["original_text"], "entities": entities_payload}
    return json.dumps(payload, ensure_ascii=False)


def _recompute_type_counters(mapping_forward: dict[str, str]) -> dict[str, int]:
    tokens_by_type: dict[str, set[str]] = {}
    for token in mapping_forward.values():
        match = _TOKEN_TYPE_RE.match(token)
        if not match:
            continue
        pii_type = match.group(1)
        tokens_by_type.setdefault(pii_type, set()).add(token)
    return {pii_type: len(tokens) for pii_type, tokens in tokens_by_type.items()}


def make_link_entities_node(llm: Any) -> Callable[[PIIState], dict[str, Any]]:
    def link_entities_node(state: PIIState) -> dict[str, Any]:
        if llm is None:
            raise RuntimeError(
                "Missing dependency or model client. Install dependencies and configure API key."
            )

        person_mentions = {e["text"] for e in state["all_entities"] if e["type"] == "PERSON"}
        if len(person_mentions) < 2:
            return {}

        raw_response = llm.invoke(
            [
                SystemMessage(content=LINKING_PROMPT),
                HumanMessage(content=_build_linking_payload(state)),
            ]
        )
        response = (
            raw_response
            if isinstance(raw_response, PIILinkingResponse)
            else _parse_linking_response_from_message(raw_response)
        )

        mapping_forward = dict(state["mapping_forward"])
        mapping_reverse = dict(state["mapping_reverse"])
        linked_pairs: list[dict[str, str]] = []

        for link in response.links:
            representative_canonical = normalize(link.representative, link.type)
            representative_token = mapping_forward.get(representative_canonical)
            if representative_token is None:
                continue

            for alias in link.aliases:
                alias_canonical = normalize(alias, link.type)
                alias_token = mapping_forward.get(alias_canonical)
                if alias_token is None or alias_token == representative_token:
                    continue

                mapping_forward[alias_canonical] = representative_token
                linked_pairs.append(
                    {
                        "type": link.type.value,
                        "from": alias,
                        "to": link.representative,
                        "from_canonical": alias_canonical,
                        "to_canonical": representative_canonical,
                        "from_token": alias_token,
                        "to_token": representative_token,
                    }
                )

        live_tokens = set(mapping_forward.values())
        mapping_reverse = {
            token: text for token, text in mapping_reverse.items() if token in live_tokens
        }

        for entity in state["all_entities"]:
            token = mapping_forward.get(entity["canonical"], entity["token"])
            current = mapping_reverse.get(token)
            if current is None or len(entity["text"]) > len(current):
                mapping_reverse[token] = entity["text"]

        return {
            "mapping_forward": mapping_forward,
            "mapping_reverse": mapping_reverse,
            "type_counters": _recompute_type_counters(mapping_forward),
            "entity_links": linked_pairs,
        }

    return link_entities_node


def make_anonymize_node() -> Callable[[PIIState], dict[str, str]]:
    def anonymize_node(state: PIIState) -> dict[str, str]:
        text = state["original_text"]
        mapping_forward = state["mapping_forward"]
        mapping_reverse = state["mapping_reverse"]

        replacements: dict[str, str] = {}
        for canonical, token in mapping_forward.items():
            replacements[canonical] = token
            if token in mapping_reverse:
                original = mapping_reverse[token]
                if original not in replacements:
                    replacements[original] = token

        for original_form in sorted(replacements, key=len, reverse=True):
            text = text.replace(original_form, replacements[original_form])

        return {"anonymized_text": text}

    return anonymize_node


def routing_fn(state: PIIState) -> Literal["process_chunk", "link_entities"]:
    if state["chunk_index"] < len(state["chunks"]):
        return "process_chunk"
    return "link_entities"
