"""
memory/episodic.py — Episodic Memory backed by a JSONL append-only log.

Episodic memory records every interaction as a timestamped episode.
Episodes are stored in a JSONL file (one JSON object per line) and can
be recalled by recency or session.

Storage schema (JSONL):
    {"session_id": str, "turn": int, "timestamp": str,
     "user": str, "assistant": str, "retrieved_from": list[str]}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.base import BaseMemory


class EpisodicMemory(BaseMemory):
    """
    Episodic Memory layer using an append-only JSONL file.

    Attributes:
        log_path: Path to the JSONL episode log file.
    """

    DEFAULT_LOG_PATH = "data/episodic_log.jsonl"

    def __init__(self, log_path: str | None = None) -> None:
        """
        Initialise EpisodicMemory.

        Args:
            log_path: Path to the JSONL log file.
                      Defaults to DEFAULT_LOG_PATH.
        """
        self.log_path = Path(log_path or self.DEFAULT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def add(
        self,
        session_id: str,
        turn: int,
        user: str,
        assistant: str,
        retrieved_from: list[str] | None = None,
    ) -> None:
        """
        Append an episode to the JSONL log.

        Args:
            session_id:     Conversation session identifier.
            turn:           Turn number within the session.
            user:           User's message text.
            assistant:      Agent's response text.
            retrieved_from: Memory layers that contributed context.

        TODO:
            - Build episode dict with timestamp (UTC ISO-8601).
            - Append JSON line to self.log_path.
        """
        # TODO: implement
        pass

    # Alias for consistency with node calls
    log = add

    def search(self, query: str, k: int = 5, **kwargs: Any) -> list[str]:
        """
        Recall recent episodes relevant to `query`.

        Simple implementation: return the last `k` episodes as formatted strings.
        Advanced implementation: embed query and rank episodes by similarity.

        Args:
            query: Natural language query (used for filtering/ranking).
            k:     Number of episodes to return.

        Returns:
            List of formatted episode strings.

        TODO:
            - Read last k lines from JSONL file.
            - Format as "[turn N] user: ... | assistant: ...".
            - (Optional) implement embedding-based ranking.
        """
        # TODO: implement
        return []

    def recall(self, session_id: str, last_n: int = 5) -> list[dict[str, Any]]:
        """
        Return the last `last_n` episodes for a given session.

        Args:
            session_id: Session to filter by.
            last_n:     Max number of episodes to return.

        Returns:
            List of episode dicts ordered oldest → newest.

        TODO:
            - Read all lines from JSONL, filter by session_id, return last last_n.
        """
        # TODO: implement
        return []

    def clear(self, session_id: str = "", **kwargs: Any) -> None:
        """
        Clear episode log.

        Args:
            session_id: If provided, remove only episodes for this session.
                        If empty, truncate the entire log.

        TODO:
            - If session_id: rewrite file excluding matching episodes.
            - If empty: truncate file.
        """
        # TODO: implement
        pass
