"""
tests/test_conflict.py — Test conflict resolution for LongTermProfileMemory.
"""
from __future__ import annotations

import pytest

from memory.long_term import LongTermProfileMemory

@pytest.fixture
def lt_mem(tmp_path):
    return LongTermProfileMemory(redis_url=None, json_path=str(tmp_path / "profile.json"))

@pytest.mark.asyncio
async def test_ltm_conflict_resolution(lt_mem: LongTermProfileMemory):
    user_id = "test_user"
    
    # 1. First save
    await lt_mem.save_fact(user_id, "diet", "chay", source="Tôi ăn chay")
    
    profile1 = await lt_mem.get_profile(user_id)
    assert profile1.get("diet") == "chay"
    
    # 2. Conflict: overwrite
    await lt_mem.save_fact(user_id, "diet", "keto", source="À nhầm, tôi ăn keto")
    
    profile2 = await lt_mem.get_profile(user_id)
    assert profile2.get("diet") == "keto"
    
    # 3. Check history
    entry = lt_mem._cache[user_id]["diet"]
    assert len(entry.history) == 1
    assert entry.history[0]["value"] == "chay"
