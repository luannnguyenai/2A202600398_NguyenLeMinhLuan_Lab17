"""
tests/test_router.py — Tests for the memory layer routing logic.

These tests verify that route_memory_layers() returns the correct set of
memory layer names for different types of user messages.

Run with:
    pytest tests/test_router.py -v
"""

from __future__ import annotations

import pytest
from agent.router import route_memory_layers, VALID_LAYERS
from agent.state import AgentState


def _make_state(user_message: str) -> AgentState:
    """Helper: return a minimal AgentState dict for testing."""
    return AgentState(
        session_id="test-session",
        turn=1,
        user_message=user_message,
        response="",
        context="",
        retrieved_from=[],
        memory_budget=3000,
        metadata={},
    )


class TestRoutingOutputValidity:
    """Ensure route_memory_layers always returns valid layer names."""

    def test_returns_list(self):
        """route_memory_layers should return a list."""
        state = _make_state("Hello, world!")
        result = route_memory_layers(state)
        assert isinstance(result, list)

    def test_all_layers_are_valid(self):
        """Every returned layer name should be in VALID_LAYERS."""
        state = _make_state("What did we discuss last week?")
        result = route_memory_layers(state)
        for layer in result:
            assert layer in VALID_LAYERS, f"Unknown layer: {layer}"

    def test_no_duplicate_layers(self):
        """route_memory_layers should not return duplicate layer names."""
        state = _make_state("Tell me about Alice.")
        result = route_memory_layers(state)
        assert len(result) == len(set(result)), "Duplicate layers returned"


class TestRoutingHeuristics:
    """Test routing decisions for specific query types (after LLM routing is implemented)."""

    @pytest.mark.skip(reason="LLM-based routing not yet implemented")
    def test_recent_context_query_includes_short_term(self):
        """
        A query about recent conversation should include short_term.

        TODO: implement after router uses LLM.
        """
        state = _make_state("What did I say earlier?")
        result = route_memory_layers(state)
        assert "short_term" in result

    @pytest.mark.skip(reason="LLM-based routing not yet implemented")
    def test_factual_query_includes_long_term(self):
        """
        A factual knowledge query should include long_term.

        TODO: implement after router uses LLM.
        """
        state = _make_state("What is the boiling point of water?")
        result = route_memory_layers(state)
        assert "long_term" in result

    @pytest.mark.skip(reason="LLM-based routing not yet implemented")
    def test_relational_query_includes_semantic(self):
        """
        A relational / 'who works at' query should include semantic.

        TODO: implement after router uses LLM.
        """
        state = _make_state("Where does Alice work?")
        result = route_memory_layers(state)
        assert "semantic" in result
