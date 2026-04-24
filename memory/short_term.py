"""
memory/short_term.py — Short-Term Memory: async sliding-window buffer.

Stores the most recent conversation turns in memory as a plain Python list.
Eviction (FIFO) is triggered when either:
  - The number of stored turns exceeds `max_turns`, OR
  - The cumulative token count exceeds `max_tokens`.

Token counting uses tiktoken (cl100k_base encoding); if tiktoken is not
installed the implementation falls back to a simple whitespace word count
(1 word ≈ 1 token, conservative approximation).

Retrieve result shape (inherited from BaseMemory):
    {"content": str, "score": float, "source": "short_term", "metadata": dict}

Where:
    content  = "{role}: {content}" formatted string
    score    = normalised recency score in (0.0, 1.0] (most recent = 1.0)
    metadata = {"role": str, "ts": float, "tokens": int}
"""

from __future__ import annotations

import time
from typing import Any

from memory.base import BaseMemory, RetrieveResult

# ---------------------------------------------------------------------------
# Optional tiktoken import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        return len(_ENCODING.encode(text))

except ImportError:  # pragma: no cover
    _ENCODING = None  # type: ignore[assignment]

    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        """Fallback: count whitespace-separated words as token approximation."""
        return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Turn entry shape (internal)
# ---------------------------------------------------------------------------

class _Turn:
    """Internal representation of a single conversation turn."""

    __slots__ = ("role", "content", "ts", "tokens")

    def __init__(self, role: str, content: str, ts: float | None = None) -> None:
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role '{role}'. Must be 'user', 'assistant', or 'system'.")
        self.role: str = role
        self.content: str = content
        self.ts: float = ts if ts is not None else time.time()
        self.tokens: int = _count_tokens(content)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation."""
        return {
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
            "tokens": self.tokens,
        }


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory(BaseMemory):
    """
    Short-Term Memory: in-process async sliding-window conversation buffer.

    The buffer holds at most `max_turns` turns and at most `max_tokens`
    cumulative tokens. Whenever a new turn is appended and either limit is
    exceeded, the oldest turn(s) are evicted (FIFO) until both constraints
    are satisfied.

    Attributes:
        max_turns:     Maximum number of turns to retain.
        max_tokens:    Maximum cumulative token count across all turns.
        _buffer:       Ordered list of _Turn objects (oldest first).
        _total_tokens: Running token count for O(1) budget checks.
    """

    SOURCE = "short_term"

    def __init__(self, max_turns: int = 10, max_tokens: int = 1500) -> None:
        """
        Initialise ShortTermMemory.

        Args:
            max_turns:  Maximum number of turns to retain (default 10).
            max_tokens: Maximum cumulative token count (default 1500).
        """
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        self.max_turns: int = max_turns
        self.max_tokens: int = max_tokens
        self._buffer: list[_Turn] = []
        self._total_tokens: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _evict_if_needed(self) -> None:
        """
        Remove the oldest turn(s) until both limits are satisfied.

        Runs in O(k) where k is the number of turns evicted (usually 1).
        Called immediately after appending a new turn.
        """
        while self._buffer and (
            len(self._buffer) > self.max_turns
            or self._total_tokens > self.max_tokens
        ):
            oldest = self._buffer.pop(0)
            self._total_tokens -= oldest.tokens

    # ------------------------------------------------------------------ #
    # BaseMemory interface                                                  #
    # ------------------------------------------------------------------ #

    async def save(
        self,
        key: str,
        value: Any,
        metadata: dict | None = None,
    ) -> None:
        """
        Append a new turn to the sliding-window buffer.

        Args:
            key:      Role of the speaker: ``"user"``, ``"assistant"``,
                      or ``"system"``. (The `key` parameter maps to *role*
                      to stay consistent with the BaseMemory interface.)
            value:    Message content string.
            metadata: Optional extra fields; currently unused by this layer
                      but accepted for interface compatibility.

        Raises:
            ValueError: If `key` is not a recognised role.

        Example::

            await stm.save("user", "Hello, I'm Alice.")
            await stm.save("assistant", "Hi Alice! How can I help?")
        """
        role = str(key)
        content = str(value)
        ts = (metadata or {}).get("ts", None)

        turn = _Turn(role=role, content=content, ts=ts)
        self._buffer.append(turn)
        self._total_tokens += turn.tokens
        self._evict_if_needed()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = 5,  # type: ignore[override]
    ) -> list[RetrieveResult]:
        """
        Return the most recent turns from the buffer.

        No semantic ranking is performed — short-term memory is ordered by
        recency only. The most recent turn receives score ``1.0``; earlier
        turns receive linearly decreasing scores.

        Args:
            query: Ignored (short-term memory is position-based, not semantic).
                   Kept for interface compatibility.
            top_k: Number of most-recent turns to return.
                   Pass ``None`` (or omit) to return all buffered turns.

        Returns:
            List of RetrieveResult dicts ordered oldest → newest.
            (Scores are assigned newest → highest so callers can rank if needed.)

        Example::

            results = await stm.retrieve("", top_k=None)
            # → [{"content": "user: Hello", "score": 0.5, ...}, ...]
        """
        if not self._buffer:
            return []

        # Decide how many turns to return
        n = len(self._buffer) if top_k is None else min(top_k, len(self._buffer))
        window = self._buffer[-n:]  # oldest-first slice of the last n turns

        total = len(window)
        results: list[RetrieveResult] = []
        for idx, turn in enumerate(window):
            # Recency score: oldest gets 1/total, newest gets 1.0
            score = (idx + 1) / total
            results.append(
                self._make_result(
                    content=f"{turn.role}: {turn.content}",
                    score=round(score, 4),
                    source=self.SOURCE,
                    metadata=turn.to_dict(),
                )
            )
        return results

    async def clear(self, key: str | None = None) -> None:
        """
        Clear the buffer.

        Args:
            key: Ignored — short-term memory is not keyed by session in this
                 in-process implementation (callers maintain one instance per
                 session). Accepted for interface compatibility.
        """
        self._buffer.clear()
        self._total_tokens = 0

    # ------------------------------------------------------------------ #
    # Extra helpers                                                        #
    # ------------------------------------------------------------------ #

    def to_messages(self) -> list[dict[str, str]]:
        """
        Return the buffer as a list of OpenAI-style message dicts.

        Suitable for direct injection into ``ChatPromptTemplate`` or the
        ``messages`` parameter of ``ChatOpenAI``.

        Returns:
            List of ``{"role": str, "content": str}`` dicts, oldest first.

        Example::

            messages = stm.to_messages()
            # → [{"role": "user", "content": "Hello"}, ...]
        """
        return [{"role": t.role, "content": t.content} for t in self._buffer]

    # ------------------------------------------------------------------ #
    # Read-only properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def turn_count(self) -> int:
        """Number of turns currently in the buffer."""
        return len(self._buffer)

    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all buffered turns."""
        return self._total_tokens

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ShortTermMemory("
            f"turns={self.turn_count}/{self.max_turns}, "
            f"tokens={self.total_tokens}/{self.max_tokens})"
        )
