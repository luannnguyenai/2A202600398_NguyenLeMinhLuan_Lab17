"""
tests/test_long_term.py — Rubric tests for LongTermProfileMemory.

These tests map directly to rubric Section 3 requirements.

Run with:
    pytest tests/test_long_term.py -v

Tests:
    1. test_save_fact_initial          — first save creates profile correctly
    2. test_save_fact_conflict_overwrite — same key + new value overwrites,
                                          old value moves to history
    3. test_delete_fact                — deleted key absent from get_profile
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from memory.long_term import LongTermProfileMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_json(tmp_path: Path) -> str:
    """Return a temporary JSON profile path inside pytest's tmp dir."""
    return str(tmp_path / "profile.json")


@pytest.fixture
def ltm(tmp_json: str) -> LongTermProfileMemory:
    """LongTermProfileMemory wired to JSON backend (no Redis needed)."""
    # Force JSON backend by passing redis_url=None and tmp json path
    return LongTermProfileMemory(redis_url=None, json_path=tmp_json)


# ---------------------------------------------------------------------------
# Test 1 — Rubric §3.a: first save creates profile correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_fact_initial(ltm: LongTermProfileMemory):
    """
    After saving a fact for the first time:
      - get_profile returns {fact_key: value}.
      - history is empty (no prior value).
      - JSON file is written to disk.
    """
    user_id = "alice"
    await ltm.save_fact(user_id, "allergy", "sữa bò", source="Alice said: I'm allergic to cow milk")

    profile = await ltm.get_profile(user_id)

    assert profile == {"allergy": "sữa bò"}, (
        f"Expected {{'allergy': 'sữa bò'}}, got {profile}"
    )

    # Internal check: history should be empty on first write
    entry = ltm._cache[user_id]["allergy"]
    assert entry.history == [], f"History should be empty on first save, got {entry.history}"
    assert entry.source == "Alice said: I'm allergic to cow milk"

    # JSON file must exist on disk
    assert Path(ltm.json_path).exists(), "JSON file should have been written to disk"


# ---------------------------------------------------------------------------
# Test 2 — Rubric §3.b: conflict → overwrite value, push old to history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_fact_conflict_overwrite(ltm: LongTermProfileMemory):
    """
    When the same fact key is saved with a different value:
      - Current value is updated to the new value.
      - Old value is pushed to history (history length == 1).
      - A third distinct value results in history length == 2.
    """
    user_id = "alice"

    # Round 1 — initial fact
    await ltm.save_fact(user_id, "allergy", "sữa bò", source="turn-1")
    profile = await ltm.get_profile(user_id)
    assert profile["allergy"] == "sữa bò"

    # Round 2 — conflict: different value
    await ltm.save_fact(user_id, "allergy", "đậu nành", source="turn-2")
    profile = await ltm.get_profile(user_id)

    assert profile == {"allergy": "đậu nành"}, (
        f"Expected updated value 'đậu nành', got {profile}"
    )

    entry = ltm._cache[user_id]["allergy"]
    assert len(entry.history) == 1, (
        f"History should have 1 entry after first conflict, got {entry.history}"
    )
    assert entry.history[0]["value"] == "sữa bò", (
        f"History should contain old value 'sữa bò', got {entry.history[0]}"
    )

    # Round 3 — another conflict
    await ltm.save_fact(user_id, "allergy", "gluten", source="turn-3")
    entry = ltm._cache[user_id]["allergy"]
    assert entry.value == "gluten"
    assert len(entry.history) == 2, (
        f"History should have 2 entries after second conflict, got {entry.history}"
    )

    # No-op: same value again should NOT grow history
    await ltm.save_fact(user_id, "allergy", "gluten", source="turn-4")
    assert len(entry.history) == 2, "No-op save should not append to history"

    # Persisted data must be consistent (reload from JSON)
    ltm2 = LongTermProfileMemory(redis_url=None, json_path=str(ltm.json_path))
    profile2 = await ltm2.get_profile(user_id)
    assert profile2["allergy"] == "gluten", (
        f"Reloaded profile should show 'gluten', got {profile2}"
    )
    entry2 = ltm2._cache[user_id]["allergy"]
    assert len(entry2.history) == 2, "Reloaded history length mismatch"


# ---------------------------------------------------------------------------
# Test 3 — Rubric §3.c: delete_fact removes key from profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_fact(ltm: LongTermProfileMemory):
    """
    After delete_fact:
      - get_profile does NOT contain the deleted key.
      - Other facts in the same profile are unaffected.
      - Deleting a non-existent key is a silent no-op.
    """
    user_id = "alice"

    await ltm.save_fact(user_id, "allergy", "sữa bò", source="turn-1")
    await ltm.save_fact(user_id, "city", "Hà Nội", source="turn-2")

    profile_before = await ltm.get_profile(user_id)
    assert "allergy" in profile_before
    assert "city" in profile_before

    # Delete the allergy fact
    await ltm.delete_fact(user_id, "allergy")

    profile_after = await ltm.get_profile(user_id)

    assert "allergy" not in profile_after, (
        f"'allergy' should be absent after delete, got {profile_after}"
    )
    assert profile_after.get("city") == "Hà Nội", (
        f"Other facts must be preserved, got {profile_after}"
    )

    # Silent no-op on missing key
    await ltm.delete_fact(user_id, "allergy")  # should not raise

    # clear() wipes the entire user profile
    await ltm.clear(key=user_id)
    profile_cleared = await ltm.get_profile(user_id)
    assert profile_cleared == {}, f"Profile should be empty after clear, got {profile_cleared}"

    # Persisted file must reflect the cleared state
    ltm2 = LongTermProfileMemory(redis_url=None, json_path=str(ltm.json_path))
    profile_reloaded = await ltm2.get_profile(user_id)
    assert profile_reloaded == {}, "Reloaded profile should be empty after clear"
