"""
memory/base.py — Abstract base class for all memory layer implementations.

Every concrete memory class must inherit from BaseMemory and implement
the three async methods below. This ensures a uniform async interface
across Short-Term, Long-Term, Episodic, and Semantic memory.

Retrieve result shape (each element in the returned list):
    {
        "content":  str,   # The raw text content
        "score":    float, # Relevance score in [0.0, 1.0]; 1.0 = most recent / most similar
        "source":   str,   # Memory layer identifier, e.g. "short_term"
        "metadata": dict,  # Layer-specific metadata (role, ts, doc_id, etc.)
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# Canonical shape of a single retrieve result.
# Kept as a plain TypedDict-style comment — avoids a hard runtime dependency
# while still giving IDE auto-complete via the type alias below.
RetrieveResult = dict  # {"content": str, "score": float, "source": str, "metadata": dict}


class BaseMemory(ABC):
    """
    Abstract base class for memory layers.

    All subclasses must implement the three async methods:
        save()     — store new information
        retrieve() — fetch relevant information for a query
        clear()    — remove stored information

    The async interface allows concrete implementations to perform
    non-blocking I/O (Redis, ChromaDB, disk) without blocking the
    LangGraph event loop.
    """

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def save(
        self,
        key: str,
        value: Any,
        metadata: dict | None = None,
    ) -> None:
        """
        Store new information in this memory layer.

        Args:
            key:      Logical key / session-id / document-id used to group
                      or look up entries later.
            value:    The content to store. Concrete type depends on the layer
                      (str message, dict triple, etc.).
            metadata: Optional key-value pairs attached to this entry
                      (e.g. {"role": "user", "turn": 3}).

        Raises:
            NotImplementedError: Must be overridden by every subclass.
        """
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrieveResult]:
        """
        Retrieve information relevant to *query*.

        Each element in the returned list must conform to:
            {
                "content":  str,
                "score":    float,  # in [0.0, 1.0]
                "source":   str,
                "metadata": dict,
            }

        Args:
            query: Natural-language query string (or session-id for exact
                   lookups that do not need semantic ranking).
            top_k: Maximum number of results to return.

        Returns:
            List of RetrieveResult dicts, ordered by descending score.

        Raises:
            NotImplementedError: Must be overridden by every subclass.
        """
        ...

    @abstractmethod
    async def clear(
        self,
        key: str | None = None,
    ) -> None:
        """
        Remove stored information.

        Args:
            key: If provided, remove only entries associated with this key
                 (e.g. a session-id). If None, wipe the entire layer.

        Raises:
            NotImplementedError: Must be overridden by every subclass.
        """
        ...

    # ------------------------------------------------------------------ #
    # Helpers available to all subclasses                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_result(
        content: str,
        score: float,
        source: str,
        metadata: dict | None = None,
    ) -> RetrieveResult:
        """
        Build a properly-shaped RetrieveResult dict.

        Convenience factory so subclasses don't have to repeat the key names.

        Args:
            content:  Raw text content.
            score:    Relevance score in [0.0, 1.0].
            source:   Memory layer identifier.
            metadata: Optional extra fields.

        Returns:
            RetrieveResult dict.
        """
        return {
            "content": content,
            "score": float(score),
            "source": source,
            "metadata": metadata or {},
        }
