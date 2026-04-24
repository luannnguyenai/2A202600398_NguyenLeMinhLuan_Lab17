"""
agent/state.py — Defines the LangGraph AgentState TypedDict.

All nodes in the graph read from and write to this shared state object.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph pipeline.

    Attributes:
        session_id:     Unique identifier for the conversation session.
        turn:           Current conversation turn index (1-based).
        user_message:   Raw text from the user's latest input.
        response:       The agent's generated response text (filled by generate node).
        context:        Aggregated context string assembled from memory retrievals.
        retrieved_from: List of memory layer names that contributed to `context`.
        memory_budget:  Remaining token budget for memory context (set by budget node).
        metadata:       Arbitrary key-value pairs for inter-node communication.
    """

    session_id: str
    turn: int
    user_message: str
    response: str
    context: str
    retrieved_from: list[str]
    memory_budget: int
    metadata: dict[str, Any]
