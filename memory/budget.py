"""
memory/budget.py — Memory Budget Manager with 4-level priority eviction.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

class Priority(enum.IntEnum):
    """
    Priority levels for context eviction. Lower number = higher priority.
    L1 is never evicted. L4 is evicted first.
    """
    L1_SYSTEM = 1
    L2_PROFILE = 2
    L3_RETRIEVAL = 3
    L4_SHORT_TERM = 4

@dataclass
class Chunk:
    """A unit of text to be packed into the context budget."""
    content: str
    priority: Priority
    tokens: int
    score: float = 0.0
    source: str = ""

class ContextBudget:
    """
    Manages token budget allocation across memory layers with priority eviction.
    """

    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4o-mini") -> None:
        self.max_tokens = max_tokens
        self.model = model
        self._encoding = None
        self._last_stats: dict[str, Any] = {"total_tokens": 0, "evicted": []}

    def _get_encoding(self) -> Any:
        if self._encoding is None and HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken, fallback to simple word count."""
        encoding = self._get_encoding()
        if encoding is not None:
            return len(encoding.encode(text))
        return max(1, len(text.split()))

    def pack(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Greedy packing of chunks into the token budget.
        
        Sorting: (priority ASC, score DESC).
        L1 chunks are never evicted.
        Logs warning if budget is near limit (>= 90%).
        """
        # Sort chunks: priority ascending (1 first), then score descending
        sorted_chunks = sorted(chunks, key=lambda c: (c.priority, -c.score))
        
        packed = []
        evicted = []
        current_tokens = 0
        
        # Guard: L1 alone must fit
        l1_tokens = sum(c.tokens for c in sorted_chunks if c.priority == Priority.L1_SYSTEM)
        if l1_tokens > self.max_tokens:
            raise ValueError(f"L1 SYSTEM chunks exceed budget: {l1_tokens} > {self.max_tokens}")

        for chunk in sorted_chunks:
            if current_tokens + chunk.tokens <= self.max_tokens:
                packed.append(chunk)
                current_tokens += chunk.tokens
            else:
                evicted.append({
                    "priority": chunk.priority.name,
                    "tokens": chunk.tokens,
                    "reason": "Exceeded budget"
                })

        # Warning when capacity >= 90%
        if current_tokens >= self.max_tokens * 0.9:
            logger.warning("context near limit, evicting L4 first")

        self._last_stats = {
            "total_tokens": current_tokens,
            "evicted": evicted
        }
        
        return packed

    def last_pack_stats(self) -> dict[str, Any]:
        """Return stats from the last packing operation."""
        return self._last_stats
