"""
memory/base.py — Abstract base class for all memory layer implementations.

Every concrete memory class must inherit from BaseMemory and implement
the abstract methods defined here. This ensures a uniform interface
across Short-Term, Long-Term, Episodic, and Semantic memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """
    Abstract base class for memory layers.

    Subclasses must implement:
        - add(...)   : Store new information.
        - search(...): Retrieve relevant information.
        - clear(...) : Remove stored information.

    The exact signatures of add/search/clear may differ per subclass
    since each memory type has different storage semantics.
    """

    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> None:
        """
        Store new information in this memory layer.

        Args:
            *args, **kwargs: Layer-specific parameters.

        TODO: Implement in each subclass.
        """
        ...

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> list[str]:
        """
        Retrieve information relevant to `query`.

        Args:
            query: The search string or embedding query.
            **kwargs: Layer-specific parameters (e.g. k=5).

        Returns:
            A list of text snippets ordered by relevance.

        TODO: Implement in each subclass.
        """
        ...

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """
        Clear stored information.

        Args:
            **kwargs: Layer-specific parameters (e.g. session_id to scope deletion).

        TODO: Implement in each subclass.
        """
        ...
