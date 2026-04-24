"""
tests/test_conflict.py — Tests for memory conflict resolution behaviour.

These tests verify that when new information contradicts previously stored
facts, the agent correctly supersedes the old value and returns the updated
information in subsequent queries.

Run with:
    pytest tests/test_conflict.py -v
"""

from __future__ import annotations

import pytest
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class TestShortTermConflict:
    """Tests for conflict resolution in Short-Term Memory."""

    def test_overwrite_on_same_session(self):
        """
        Adding a new value for the same key/session should supersede the old one.

        TODO:
            - Add message A to session.
            - Add message B (contradicting A) to same session.
            - search() should return B as the most recent context.
        """
        # TODO: implement when ShortTermMemory.add() is done
        stm = ShortTermMemory()
        pytest.skip("ShortTermMemory.add() not yet implemented")

    def test_window_eviction_removes_old_context(self):
        """
        When the sliding window is full, the oldest messages should be evicted.

        TODO:
            - Fill a session to window_size + 1 messages.
            - Assert that the first message is no longer in the window.
        """
        pytest.skip("ShortTermMemory.add() not yet implemented")


class TestLongTermConflict:
    """Tests for conflict resolution in Long-Term Memory."""

    def test_updated_fact_ranks_higher_in_search(self):
        """
        A more recently stored fact should rank higher than an older contradicting one.

        TODO:
            - Store "Alice lives in Berlin".
            - Store "Alice lives in Paris".
            - search("Where does Alice live?") should return Paris-related result first.
        """
        # TODO: implement when LongTermMemory is done
        pytest.skip("LongTermMemory not yet implemented")

    def test_explicit_delete_before_update(self):
        """
        After deleting a stale document, only the new fact should be retrievable.

        TODO:
            - Store fact F1 with a known doc_id.
            - Delete doc_id.
            - Store updated fact F2.
            - Assert only F2 is returned by search().
        """
        pytest.skip("LongTermMemory not yet implemented")
