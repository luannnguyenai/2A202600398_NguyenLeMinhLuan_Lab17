"""
agent/nodes.py — LangGraph node functions.

Each function takes an AgentState and returns a partial AgentState dict
with the fields it modifies. LangGraph merges the returned dict into the
running state automatically.

Node execution order (defined in graph.py):
  budget_node → retrieve_node → generate_node → store_node
"""

from __future__ import annotations

import os
from typing import Any

from agent.state import AgentState
from agent.router import route_memory_layers, should_store_to_long_term
from memory.budget import MemoryBudgetManager
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory


# ---------------------------------------------------------------------------
# Lazy singletons — initialised on first use to avoid import-time side effects
# ---------------------------------------------------------------------------

_short_term: ShortTermMemory | None = None
_long_term: LongTermMemory | None = None
_episodic: EpisodicMemory | None = None
_semantic: SemanticMemory | None = None
_budget_manager: MemoryBudgetManager | None = None


def _get_memories() -> tuple[
    ShortTermMemory, LongTermMemory, EpisodicMemory, SemanticMemory, MemoryBudgetManager
]:
    """Initialise and return memory layer singletons."""
    global _short_term, _long_term, _episodic, _semantic, _budget_manager
    if _short_term is None:
        _short_term = ShortTermMemory()
        _long_term = LongTermMemory()
        _episodic = EpisodicMemory()
        _semantic = SemanticMemory()
        _budget_manager = MemoryBudgetManager(
            max_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
        )
    return _short_term, _long_term, _episodic, _semantic, _budget_manager  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Node: budget_node
# ---------------------------------------------------------------------------


def budget_node(state: AgentState) -> dict[str, Any]:
    """
    Calculate available token budget for memory context.

    Reads MAX_CONTEXT_TOKENS from environment, subtracts estimated tokens
    used by system prompt and user message, and writes the remainder to
    state.memory_budget.

    Args:
        state: Current AgentState.

    Returns:
        Dict with key "memory_budget" (int).

    TODO:
        - Use tiktoken to count tokens in user_message + system prompt.
        - Subtract from MAX_CONTEXT_TOKENS.
        - Write result to state["memory_budget"].
    """
    _, _, _, _, budget_manager = _get_memories()
    # TODO: implement actual token counting
    budget = budget_manager.compute_budget(
        user_message=state["user_message"],
        system_overhead=500,  # rough estimate
    )
    return {"memory_budget": budget}


# ---------------------------------------------------------------------------
# Node: retrieve_node
# ---------------------------------------------------------------------------


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """
    Query memory layers and assemble context string.

    Uses router.route_memory_layers() to decide which layers to query,
    then fetches results from each, truncates to memory_budget tokens,
    and assembles into state.context.

    Args:
        state: Current AgentState with user_message and memory_budget.

    Returns:
        Dict with keys "context" (str) and "retrieved_from" (list[str]).

    TODO:
        - Call route_memory_layers(state) to get target layers.
        - Query each layer: short_term.get(), long_term.search(),
          episodic.recall(), semantic.query().
        - Concatenate results, respecting memory_budget.
        - Populate retrieved_from with layer names that returned content.
    """
    short_term, long_term, episodic, semantic, _ = _get_memories()
    layers = route_memory_layers(state)

    context_parts: list[str] = []
    retrieved_from: list[str] = []

    # TODO: implement actual retrieval per layer
    for layer in layers:
        pass  # placeholder

    context = "\n\n".join(context_parts) if context_parts else "(no memory context)"
    return {"context": context, "retrieved_from": retrieved_from}


# ---------------------------------------------------------------------------
# Node: generate_node
# ---------------------------------------------------------------------------


def generate_node(state: AgentState) -> dict[str, Any]:
    """
    Generate the agent's response using the LLM.

    Combines the assembled context with the user message via CHAT_PROMPT
    and calls the configured ChatOpenAI model.

    Args:
        state: Current AgentState with context and user_message.

    Returns:
        Dict with key "response" (str).

    TODO:
        - Import CHAT_PROMPT from agent.prompt.
        - Instantiate ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini")).
        - Format prompt with context and user_message.
        - Invoke LLM and extract .content from the result.
    """
    # Stub: echo the user message
    return {"response": f"[STUB] Echo: {state['user_message']}"}


# ---------------------------------------------------------------------------
# Node: store_node
# ---------------------------------------------------------------------------


def store_node(state: AgentState) -> dict[str, Any]:
    """
    Persist the current exchange to appropriate memory layers.

    Writes to:
      - short_term: always (sliding window of recent turns)
      - long_term:  if should_store_to_long_term(state) is True
      - episodic:   always (full interaction log)

    Args:
        state: Completed AgentState after generate_node has run.

    Returns:
        Empty dict (no state fields modified).

    TODO:
        - Call short_term.add(session_id, user_message, response).
        - Conditionally call long_term.add(user_message, response).
        - Call episodic.log(session_id, turn, user_message, response).
    """
    short_term, long_term, episodic, _, _ = _get_memories()
    # TODO: implement storage calls
    return {}
