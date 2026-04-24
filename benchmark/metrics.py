"""
benchmark/metrics.py — Scoring functions for the benchmark evaluation.

Metrics computed per conversation turn:
  - keyword_hit_rate:  Fraction of expected_keywords present in response.
  - exact_match:       1 if all keywords are present, 0 otherwise.
  - response_latency:  Wall-clock time for agent to respond (seconds).

Aggregate metrics (per conversation / overall):
  - mean_keyword_hit_rate
  - exact_match_rate
  - mean_latency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnResult:
    """Result for a single conversation turn."""

    turn_index: int
    user_message: str
    response: str
    expected_keywords: list[str]
    retrieved_from: list[str]
    latency_seconds: float
    keyword_hits: list[str] = field(default_factory=list)
    keyword_misses: list[str] = field(default_factory=list)

    @property
    def keyword_hit_rate(self) -> float:
        """Fraction of expected keywords present in the response."""
        if not self.expected_keywords:
            return 1.0  # No keywords required → perfect score
        return len(self.keyword_hits) / len(self.expected_keywords)

    @property
    def exact_match(self) -> bool:
        """True if all expected keywords are present."""
        return len(self.keyword_misses) == 0


@dataclass
class ConversationResult:
    """Aggregated result for a full conversation."""

    conversation_id: str
    conversation_name: str
    tags: list[str]
    turn_results: list[TurnResult] = field(default_factory=list)

    @property
    def mean_keyword_hit_rate(self) -> float:
        """Average keyword hit rate across all turns."""
        if not self.turn_results:
            return 0.0
        return sum(t.keyword_hit_rate for t in self.turn_results) / len(self.turn_results)

    @property
    def exact_match_rate(self) -> float:
        """Fraction of turns where all keywords were present."""
        if not self.turn_results:
            return 0.0
        return sum(1 for t in self.turn_results if t.exact_match) / len(self.turn_results)

    @property
    def mean_latency(self) -> float:
        """Average response latency in seconds."""
        if not self.turn_results:
            return 0.0
        return sum(t.latency_seconds for t in self.turn_results) / len(self.turn_results)


def score_turn(
    response: str,
    expected_keywords: list[str],
) -> tuple[list[str], list[str]]:
    """
    Check which expected keywords are present in the response.

    Case-insensitive substring match.

    Args:
        response:          Agent's response text.
        expected_keywords: List of keywords that should appear.

    Returns:
        Tuple of (keyword_hits, keyword_misses).

    TODO:
        - Optionally add lemmatisation / synonym matching for robustness.
    """
    response_lower = response.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in response_lower]
    misses = [kw for kw in expected_keywords if kw.lower() not in response_lower]
    return hits, misses


def compute_metrics(
    results: list[ConversationResult],
) -> dict[str, Any]:
    """
    Compute aggregate metrics across all conversations.

    Args:
        results: List of ConversationResult objects from the benchmark run.

    Returns:
        Dict with keys:
            overall_keyword_hit_rate (float)
            overall_exact_match_rate (float)
            mean_latency_seconds     (float)
            per_tag                  (dict[str, dict])  — metrics per capability tag
            total_conversations      (int)
            total_turns              (int)

    TODO:
        - Aggregate per-tag metrics.
        - Add per-conversation breakdown.
    """
    if not results:
        return {}

    all_turns = [t for conv in results for t in conv.turn_results]
    total_turns = len(all_turns)

    overall_khr = sum(t.keyword_hit_rate for t in all_turns) / total_turns if total_turns else 0.0
    overall_emr = sum(1 for t in all_turns if t.exact_match) / total_turns if total_turns else 0.0
    mean_latency = sum(t.latency_seconds for t in all_turns) / total_turns if total_turns else 0.0

    # TODO: implement per-tag aggregation
    per_tag: dict[str, dict] = {}

    return {
        "overall_keyword_hit_rate": round(overall_khr, 4),
        "overall_exact_match_rate": round(overall_emr, 4),
        "mean_latency_seconds": round(mean_latency, 4),
        "per_tag": per_tag,
        "total_conversations": len(results),
        "total_turns": total_turns,
    }
