"""
tests/test_budget.py — Tests for ContextBudget priority eviction.
"""
from __future__ import annotations

import pytest

from memory.budget import ContextBudget, Priority, Chunk

def test_pack_under_budget():
    """All chunks fit perfectly without exceeding max_tokens."""
    budget = ContextBudget(max_tokens=100)
    chunks = [
        Chunk(content="System", priority=Priority.L1_SYSTEM, tokens=20, score=1.0),
        Chunk(content="Profile", priority=Priority.L2_PROFILE, tokens=20, score=0.8),
        Chunk(content="Recent", priority=Priority.L4_SHORT_TERM, tokens=30, score=1.0)
    ]
    packed = budget.pack(chunks)
    
    assert len(packed) == 3
    stats = budget.last_pack_stats()
    assert stats["total_tokens"] == 70
    assert len(stats["evicted"]) == 0

def test_pack_over_budget_evicts_l4_before_l3():
    """Eviction starts from the lowest priority (highest number) first."""
    budget = ContextBudget(max_tokens=60)
    
    # Total tokens = 70. 
    # Should evict L4 (20 tokens). 
    # L1 (20) + L3 (30) = 50 tokens (fits).
    chunks = [
        Chunk(content="System", priority=Priority.L1_SYSTEM, tokens=20),
        Chunk(content="Retrieval", priority=Priority.L3_RETRIEVAL, tokens=30),
        Chunk(content="Recent Chat", priority=Priority.L4_SHORT_TERM, tokens=20)
    ]
    packed = budget.pack(chunks)
    
    # Expect L1 and L3 to be packed
    assert len(packed) == 2
    assert packed[0].priority == Priority.L1_SYSTEM
    assert packed[1].priority == Priority.L3_RETRIEVAL
    
    stats = budget.last_pack_stats()
    assert stats["total_tokens"] == 50
    assert len(stats["evicted"]) == 1
    assert stats["evicted"][0]["priority"] == "L4_SHORT_TERM"

def test_l1_exceeds_budget():
    """If L1 alone exceeds the total budget, raise ValueError."""
    budget = ContextBudget(max_tokens=10)
    chunks = [
        Chunk(content="Huge System Prompt", priority=Priority.L1_SYSTEM, tokens=20)
    ]
    
    with pytest.raises(ValueError, match="L1 SYSTEM chunks exceed budget"):
        budget.pack(chunks)
