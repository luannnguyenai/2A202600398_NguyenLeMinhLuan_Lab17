"""
tests/test_short_term.py — Unit tests for ShortTermMemory.

Run with:
    pytest tests/test_short_term.py -v

Tests:
    1. test_append_and_retrieve_order  — appended turns come back in correct order
    2. test_evict_on_max_turns         — buffer does not exceed max_turns
    3. test_evict_on_max_tokens        — buffer does not exceed max_tokens
"""

from __future__ import annotations

import asyncio
import time

import pytest
import pytest_asyncio

from memory.short_term import ShortTermMemory, _count_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fill(stm: ShortTermMemory, pairs: list[tuple[str, str]]) -> None:
    """Save a list of (role, content) pairs into *stm*."""
    for role, content in pairs:
        await stm.save(role, content)


# ---------------------------------------------------------------------------
# Test 1 — append + retrieve in correct order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_and_retrieve_order():
    """
    Turns must be returned oldest-first by retrieve(), preserving
    insertion order and correct role/content values.
    """
    stm = ShortTermMemory(max_turns=10, max_tokens=4000)

    await _fill(stm, [
        ("user",      "Hello, my name is Alice."),
        ("assistant", "Hi Alice! Nice to meet you."),
        ("user",      "What is the capital of France?"),
        ("assistant", "The capital of France is Paris."),
    ])

    results = await stm.retrieve("", top_k=None)

    # Correct count
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    # Oldest turn is first
    assert results[0]["metadata"]["role"] == "user"
    assert "Alice" in results[0]["content"]

    # Newest turn is last
    assert results[-1]["metadata"]["role"] == "assistant"
    assert "Paris" in results[-1]["content"]

    # Scores increase monotonically (oldest → lowest, newest → 1.0)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores), "Scores should be non-decreasing"
    assert results[-1]["score"] == pytest.approx(1.0)

    # to_messages() preserves order and OpenAI message shape
    messages = stm.to_messages()
    assert messages[0] == {"role": "user", "content": "Hello, my name is Alice."}
    assert messages[-1] == {"role": "assistant", "content": "The capital of France is Paris."}

    # source field
    for r in results:
        assert r["source"] == "short_term"

    # Every result has the required RetrieveResult keys
    for r in results:
        assert "content" in r
        assert "score" in r
        assert "source" in r
        assert "metadata" in r


# ---------------------------------------------------------------------------
# Test 2 — evict when max_turns is exceeded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evict_on_max_turns():
    """
    When more turns than max_turns are saved, the oldest turns must be
    evicted so that turn_count never exceeds max_turns.
    """
    MAX = 3
    stm = ShortTermMemory(max_turns=MAX, max_tokens=99_999)

    await _fill(stm, [
        ("user",      "Turn 1"),
        ("assistant", "Turn 2"),
        ("user",      "Turn 3"),
    ])
    assert stm.turn_count == MAX

    # Adding a 4th turn should evict "Turn 1"
    await stm.save("assistant", "Turn 4")

    assert stm.turn_count == MAX, (
        f"Buffer should hold exactly {MAX} turns, got {stm.turn_count}"
    )

    results = await stm.retrieve("", top_k=None)
    contents = [r["content"] for r in results]

    # "Turn 1" must have been evicted
    assert not any("Turn 1" in c for c in contents), (
        "Oldest turn should have been evicted but is still present"
    )
    # "Turn 4" (newest) must be present
    assert any("Turn 4" in c for c in contents), (
        "Newest turn should be in the buffer"
    )

    # Adding many more turns should never exceed MAX
    for i in range(5, 20):
        await stm.save("user", f"Turn {i}")
        assert stm.turn_count <= MAX, (
            f"turn_count exceeded max_turns at turn {i}: {stm.turn_count}"
        )


# ---------------------------------------------------------------------------
# Test 3 — evict when max_tokens is exceeded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evict_on_max_tokens():
    """
    When the cumulative token count exceeds max_tokens, turns must be
    evicted (oldest first) until the token budget is back within the limit.
    """
    # A short message to get a predictable token count
    msg = "hello world"
    msg_tokens = _count_tokens(msg)
    assert msg_tokens >= 1, "token counter must return at least 1"

    # Allow exactly 2 messages worth of tokens
    max_tokens = msg_tokens * 2
    stm = ShortTermMemory(max_turns=999, max_tokens=max_tokens)

    # Save 2 messages — should fit exactly
    await stm.save("user", msg)
    await stm.save("assistant", msg)
    assert stm.turn_count == 2
    assert stm.total_tokens <= max_tokens

    # Save a 3rd message — should evict the oldest to stay within budget
    await stm.save("user", msg)

    assert stm.total_tokens <= max_tokens, (
        f"total_tokens={stm.total_tokens} exceeds max_tokens={max_tokens}"
    )
    assert stm.turn_count <= 2, (
        f"turn_count={stm.turn_count} should be at most 2 after eviction"
    )

    # The newest message must still be present
    results = await stm.retrieve("", top_k=None)
    assert results, "Buffer must not be empty after eviction"
    assert results[-1]["metadata"]["role"] == "user"
    assert msg in results[-1]["content"]

    # Verify clear() resets everything
    await stm.clear()
    assert stm.turn_count == 0
    assert stm.total_tokens == 0
    assert await stm.retrieve("") == []
