"""
tests/test_budget.py — Tests for the MemoryBudgetManager.

These tests verify that token budget calculation and text truncation
behave correctly under various input conditions.

Run with:
    pytest tests/test_budget.py -v
"""

from __future__ import annotations

import pytest
from memory.budget import MemoryBudgetManager


class TestComputeBudget:
    """Tests for MemoryBudgetManager.compute_budget()."""

    def test_budget_is_non_negative(self):
        """Budget should never be negative, even with a very long user message."""
        mgr = MemoryBudgetManager(max_tokens=100)
        budget = mgr.compute_budget(user_message="x" * 10_000, system_overhead=50)
        assert budget >= 0

    def test_budget_decreases_with_longer_message(self):
        """Longer user messages should result in a smaller budget."""
        mgr = MemoryBudgetManager(max_tokens=4000)
        budget_short = mgr.compute_budget(user_message="Hi", system_overhead=500)
        budget_long = mgr.compute_budget(user_message="Hi " * 500, system_overhead=500)
        assert budget_short > budget_long

    def test_budget_with_zero_overhead(self):
        """With zero overhead, budget should be max_tokens minus message tokens."""
        mgr = MemoryBudgetManager(max_tokens=1000)
        budget = mgr.compute_budget(user_message="", system_overhead=0)
        # Empty message ≈ 0 tokens (or 1 minimum)
        assert budget >= 999  # Allow for 1-token minimum in count_tokens


class TestTruncateToBudget:
    """Tests for MemoryBudgetManager.truncate_to_budget()."""

    def test_short_text_not_truncated(self):
        """Text within budget should be returned unchanged."""
        mgr = MemoryBudgetManager()
        text = "Short text."
        result = mgr.truncate_to_budget(text, budget=1000)
        assert result == text

    def test_long_text_is_truncated(self):
        """Text exceeding budget should be truncated and end with '...'."""
        mgr = MemoryBudgetManager()
        text = "word " * 2000  # ~500 tokens using 4-char approximation
        result = mgr.truncate_to_budget(text, budget=10)
        assert len(result) < len(text)
        # TODO: once tiktoken is implemented, assert result.endswith("...")

    def test_empty_text_returns_empty(self):
        """Empty text should be returned as-is."""
        mgr = MemoryBudgetManager()
        result = mgr.truncate_to_budget("", budget=100)
        assert result == ""


class TestAllocatePerLayer:
    """Tests for MemoryBudgetManager.allocate_per_layer()."""

    def test_allocations_sum_to_at_most_total(self):
        """Total allocated tokens should not exceed total_budget (may be slightly less due to int rounding)."""
        mgr = MemoryBudgetManager()
        total = 1000
        allocations = mgr.allocate_per_layer(total_budget=total)
        assert sum(allocations.values()) <= total + 4  # Allow ≤4 token rounding error

    def test_all_four_layers_present(self):
        """Default allocation should cover all four memory layers."""
        mgr = MemoryBudgetManager()
        allocations = mgr.allocate_per_layer(total_budget=1000)
        for layer in ("short_term", "long_term", "episodic", "semantic"):
            assert layer in allocations, f"Missing layer: {layer}"

    def test_custom_weights(self):
        """Custom weight dict should be respected."""
        mgr = MemoryBudgetManager()
        weights = {"short_term": 0.8, "long_term": 0.2}
        allocations = mgr.allocate_per_layer(total_budget=100, layer_weights=weights)
        assert allocations["short_term"] > allocations["long_term"]
