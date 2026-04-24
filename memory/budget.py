"""
memory/budget.py — Memory Budget Manager.

Manages the token budget available for memory context so that the total
prompt (system prompt + memory context + user message) stays within
the model's context window.

Uses tiktoken for accurate token counting.
"""

from __future__ import annotations

import os
from typing import Any


class MemoryBudgetManager:
    """
    Manages token budget allocation across memory layers.

    Attributes:
        max_tokens:    Maximum total tokens allowed in the context window.
        _encoding:     Lazy-initialised tiktoken encoding object.
        _model_name:   Name of the model (used to select tiktoken encoding).
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        Initialise the budget manager.

        Args:
            max_tokens:  Token budget. Defaults to MAX_CONTEXT_TOKENS env var.
            model_name:  Model name for tiktoken encoding selection.
                         Defaults to MODEL env var.
        """
        self.max_tokens = max_tokens or int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
        self._model_name = model_name or os.getenv("MODEL", "gpt-4o-mini")
        self._encoding = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoding(self):
        """
        Return (and lazily initialise) the tiktoken encoding.

        TODO:
            - Import tiktoken.
            - Use tiktoken.encoding_for_model(self._model_name) with fallback
              to tiktoken.get_encoding("cl100k_base").
        """
        if self._encoding is None:
            # TODO: initialise tiktoken encoding
            pass
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text string.

        Args:
            text: Input text.

        Returns:
            Token count as an integer.

        TODO:
            - Use self._get_encoding().encode(text) and return len().
        """
        # TODO: implement with tiktoken
        # Rough estimate fallback: 4 chars ≈ 1 token
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_budget(self, user_message: str, system_overhead: int = 500) -> int:
        """
        Calculate available token budget for memory context.

        Budget = max_tokens - system_overhead - count_tokens(user_message)

        Args:
            user_message:     The user's input text.
            system_overhead:  Estimated tokens used by system prompt template.

        Returns:
            Available token count for memory context (minimum 0).

        TODO:
            - Use self.count_tokens() for accurate counting.
        """
        used = system_overhead + self.count_tokens(user_message)
        return max(0, self.max_tokens - used)

    def truncate_to_budget(self, text: str, budget: int) -> str:
        """
        Truncate `text` so that it fits within `budget` tokens.

        Args:
            text:   The text to truncate.
            budget: Maximum allowed tokens.

        Returns:
            Truncated text string. May end mid-sentence; consider adding "...".

        TODO:
            - Encode text with tiktoken.
            - Slice to budget tokens.
            - Decode back to string.
        """
        # TODO: implement with tiktoken
        # Rough fallback: 1 token ≈ 4 chars
        max_chars = budget * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def allocate_per_layer(
        self,
        total_budget: int,
        layer_weights: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """
        Distribute token budget across memory layers by weight.

        Default weights: short_term=0.4, long_term=0.3, episodic=0.2, semantic=0.1

        Args:
            total_budget:  Total tokens available for memory context.
            layer_weights: Optional override for per-layer proportions.
                           Values should sum to 1.0.

        Returns:
            Dict mapping layer name → token count.

        TODO:
            - Normalise weights if they don't sum to 1.
            - Distribute total_budget proportionally.
            - Ensure integer values.
        """
        defaults = {
            "short_term": 0.4,
            "long_term": 0.3,
            "episodic": 0.2,
            "semantic": 0.1,
        }
        weights = layer_weights or defaults
        # TODO: implement proper allocation
        return {layer: int(total_budget * w) for layer, w in weights.items()}
