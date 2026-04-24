"""
agent/router.py — Memory layer routing logic.

The router inspects the incoming AgentState and decides which memory
layers should be queried for the current user message.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompt import ROUTING_PROMPT

# All valid memory layer names
VALID_LAYERS: frozenset[str] = frozenset(
    ["short_term", "long_term", "episodic", "semantic"]
)


def route_memory_layers(state: AgentState) -> list[str]:
    """
    Determine which memory layers to query for a given user message.

    Strategy (TODO — replace with LLM-based routing):
      1. Always include "short_term" (recent context is nearly always useful).
      2. Use an LLM call with ROUTING_PROMPT to select additional layers.
      3. Fallback to all layers if the LLM response cannot be parsed.

    Args:
        state: Current AgentState with at minimum `user_message` populated.

    Returns:
        A list of memory layer names to query, e.g. ["short_term", "semantic"].

    TODO:
        - Instantiate ChatOpenAI from env (MODEL env var).
        - Invoke ROUTING_PROMPT | llm and parse JSON response.
        - Validate layer names against VALID_LAYERS.
        - Handle API errors gracefully (fallback to all layers).
    """
    # Stub: always route to all layers
    return list(VALID_LAYERS)


def should_store_to_long_term(state: AgentState) -> bool:
    """
    Decide whether the current exchange should be persisted to long-term memory.

    Heuristic (TODO — implement proper logic):
      - Store if the user message contains a factual assertion.
      - Store if the response length exceeds a threshold (information-dense).

    Args:
        state: Completed AgentState after generate node has run.

    Returns:
        True if the exchange should be stored in long-term memory.

    TODO:
        - Implement keyword / NLI heuristic.
    """
    # Stub: always store
    return True
