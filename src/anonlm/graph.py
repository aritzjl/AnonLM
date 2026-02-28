"""LangGraph builder for the AnonLM anonymization pipeline."""

from __future__ import annotations

from typing import Any

from langchain_core.exceptions import OutputParserException
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, default_retry_on

from anonlm.config import AnonLMConfig
from anonlm.nodes import make_anonymize_node, make_prepare_node, make_process_chunk_node, routing_fn
from anonlm.state import PIIState


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, OutputParserException):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 425, 429}:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status in {408, 409, 425, 429}:
        return True

    return default_retry_on(exc)


def create_graph(config: AnonLMConfig, llm: Any):
    process_chunk_retry = RetryPolicy(
        initial_interval=config.retry_initial_interval,
        backoff_factor=config.retry_backoff_factor,
        max_interval=config.retry_max_interval,
        max_attempts=config.retry_max_attempts,
        jitter=True,
        retry_on=_should_retry,
    )

    graph = StateGraph(PIIState)
    graph.add_node("prepare", make_prepare_node(config))
    graph.add_node("process_chunk", make_process_chunk_node(llm), retry_policy=process_chunk_retry)
    graph.add_node("anonymize", make_anonymize_node())

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "process_chunk")
    graph.add_conditional_edges("process_chunk", routing_fn)
    graph.add_edge("anonymize", END)

    return graph.compile()
