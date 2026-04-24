"""
memory/short_term.py — Short-Term Memory backed by Redis.

Short-term memory stores the most recent N conversation turns for a
session using a Redis list. It acts as a sliding-window context buffer.

Storage schema (Redis):
    Key   : f"stm:{session_id}"
    Type  : Redis List (RPUSH / LTRIM)
    Value : JSON-serialised {"role": str, "content": str}
"""

from __future__ import annotations

import json
import os
from typing import Any

from memory.base import BaseMemory


class ShortTermMemory(BaseMemory):
    """
    Short-Term Memory layer using Redis as the backing store.

    Attributes:
        redis_url:   Redis connection URL from env REDIS_URL.
        window_size: Maximum number of messages to retain per session.
        _client:     Lazy-initialised Redis client.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        window_size: int = 10,
    ) -> None:
        """
        Initialise ShortTermMemory.

        Args:
            redis_url:   Redis URL. Defaults to REDIS_URL env var.
            window_size: Max messages to retain (older ones are evicted).
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.window_size = window_size
        self._client = None  # Lazy-initialised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """
        Return (and lazily create) the Redis client.

        TODO:
            - Import redis and call redis.from_url(self.redis_url).
            - Cache in self._client.
        """
        if self._client is None:
            # TODO: initialise redis client
            pass
        return self._client

    def _make_key(self, session_id: str) -> str:
        """Return the Redis key for a given session."""
        return f"stm:{session_id}"

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def add(self, session_id: str, role: str, content: str) -> None:
        """
        Append a message to the session's short-term memory list.

        Keeps only the latest `window_size` messages (sliding window).

        Args:
            session_id: Conversation session identifier.
            role:       "user" or "assistant".
            content:    Message text.

        TODO:
            - Serialise to JSON and RPUSH to Redis key.
            - LTRIM to window_size * 2 (user + assistant pairs).
        """
        # TODO: implement
        pass

    def search(self, query: str, session_id: str = "", **kwargs: Any) -> list[str]:
        """
        Return all messages in the session window (no semantic search).

        Args:
            query:      Ignored for short-term memory (returns full window).
            session_id: Session to retrieve messages from.

        Returns:
            List of formatted strings, e.g. ["user: Hi", "assistant: Hello"].

        TODO:
            - LRANGE key 0 -1 from Redis.
            - Deserialise JSON and format as "{role}: {content}".
        """
        # TODO: implement
        return []

    def get(self, session_id: str) -> list[dict[str, str]]:
        """
        Return raw message dicts for a session.

        Args:
            session_id: Session to retrieve.

        Returns:
            List of {"role": str, "content": str} dicts.

        TODO:
            - LRANGE and deserialise.
        """
        # TODO: implement
        return []

    def clear(self, session_id: str = "", **kwargs: Any) -> None:
        """
        Delete all messages for a session.

        Args:
            session_id: Session to clear.

        TODO:
            - DEL key from Redis.
        """
        # TODO: implement
        pass
