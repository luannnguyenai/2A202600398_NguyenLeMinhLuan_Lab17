"""
memory/episodic.py — Episodic Memory: append-only JSONL episode log.

Each episode records a completed task interaction with outcome, summary,
lesson learned, and searchable tags. Only episodes with a clear outcome
("success" | "failure" | "neutral") are persisted — chit-chat is excluded.

Storage:
    Append-only JSONL file (one JSON object per line).
    In-process list mirrors the file for fast retrieval without re-reading.

Retrieval scoring:
    keyword_overlap(query, task + summary + lesson + tags)
    + 0.1 bonus if outcome == "success" and lesson is not None
    Normalised to [0.0, 1.0].

Episode schema:
    {
        "id":       "ep_<uuid4>",
        "user_id":  str,
        "ts":       float,           # UTC epoch
        "task":     str,             # what was attempted
        "outcome":  "success" | "failure" | "neutral",
        "summary":  str,             # 1-2 sentence description
        "lesson":   str | None,      # what was learned (or None)
        "tags":     list[str]
    }
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from memory.base import BaseMemory, RetrieveResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OUTCOMES: frozenset[str] = frozenset({"success", "failure", "neutral"})

# Simple English + Vietnamese stopwords to ignore during keyword scoring
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "for",
    "with", "by", "from", "and", "or", "but", "not", "so", "if", "that",
    "this", "it", "its", "i", "you", "he", "she", "we", "they", "what",
    "how", "when", "where", "who", "which",
    # Vietnamese common particles
    "và", "của", "là", "có", "không", "được", "để", "trong", "với",
    "cho", "tôi", "bạn", "này", "đó", "một", "các", "những", "khi",
    "như", "về", "từ", "ra", "vào", "đã", "sẽ", "đang", "thì", "mà",
})

_BONUS_SUCCESS_WITH_LESSON = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """
    Lowercase and split text into tokens, removing stopwords and
    short tokens (length < 2).
    """
    tokens = set()
    for word in text.lower().split():
        # Strip common punctuation
        word = word.strip(".,;:!?\"'()[]{}")
        if len(word) >= 2 and word not in _STOPWORDS:
            tokens.add(word)
    return tokens


def _score_episode(query_tokens: set[str], ep: dict[str, Any]) -> float:
    """
    Compute a relevance score for *ep* given *query_tokens*.

    Corpus = task + summary + (lesson or "") + space-joined tags.
    Base   = |query ∩ corpus| / |query|  (precision over query tokens).
    Bonus  = +0.10 if outcome == "success" and lesson is not None.

    The raw score is returned **without clamping** to 1.0 so that the
    bonus is always distinguishable from the base score. Callers are
    responsible for clamping when they need a [0, 1] output.

    Returns 0.0 if query_tokens is empty.
    """
    if not query_tokens:
        return 0.0

    corpus = " ".join(filter(None, [
        ep.get("task", ""),
        ep.get("summary", ""),
        ep.get("lesson") or "",
        " ".join(ep.get("tags", [])),
    ]))
    corpus_tokens = _tokenize(corpus)

    overlap = len(query_tokens & corpus_tokens)
    base_score = overlap / len(query_tokens)

    bonus = (
        _BONUS_SUCCESS_WITH_LESSON
        if ep.get("outcome") == "success" and ep.get("lesson")
        else 0.0
    )

    # Return raw (possibly > 1.0); clamping is done at the call-site
    return base_score + bonus


def _make_episode(ep: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise and validate an episode dict.

    Auto-fills:
        id  → "ep_<uuid4>" if missing
        ts  → current UTC epoch if missing

    Raises:
        ValueError: if `outcome` is missing or not in VALID_OUTCOMES.
    """
    outcome = ep.get("outcome")
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"Episode outcome must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}. "
            "Chit-chat without a clear outcome must not be stored."
        )

    return {
        "id": ep.get("id") or f"ep_{uuid.uuid4().hex}",
        "user_id": str(ep.get("user_id", "default")),
        "ts": float(ep.get("ts") or time.time()),
        "task": str(ep.get("task", "")),
        "outcome": outcome,
        "summary": str(ep.get("summary", "")),
        "lesson": ep.get("lesson") or None,
        "tags": [str(t) for t in ep.get("tags", [])],
    }


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory(BaseMemory):
    """
    Episodic Memory — append-only JSONL log with keyword-based retrieval.

    The file on disk is the source of truth; an in-process list mirrors it
    after the first load so retrieval is O(n) in RAM without repeated I/O.

    Attributes:
        log_path:   Path to the JSONL episode log file.
        _episodes:  In-process list of episode dicts (loaded lazily).
        _loaded:    Whether the file has been read into _episodes yet.
    """

    SOURCE = "episodic"

    def __init__(self, log_path: str = "data/episodic_log.jsonl") -> None:
        """
        Initialise EpisodicMemory.

        Args:
            log_path: Path to the JSONL log. Parent directories are created
                      automatically on first write.
        """
        self.log_path = Path(log_path)
        self._episodes: list[dict[str, Any]] = []
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    # File I/O                                                             #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """
        Read the JSONL file into the in-process cache (once).

        Silently skips malformed lines (logs a warning per bad line).
        """
        if self._loaded:
            return
        self._episodes = []
        if self.log_path.exists():
            for lineno, line in enumerate(
                self.log_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Skipping malformed JSONL line %d in %s", lineno, self.log_path
                    )
        self._loaded = True

    def _append_to_file(self, ep: dict[str, Any]) -> None:
        """Append a single episode JSON line to the JSONL file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")

    def _rewrite_file(self, episodes: list[dict[str, Any]]) -> None:
        """Atomically rewrite the JSONL file with a new episode list."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.log_path.with_suffix(".tmp")
        lines = "\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes)
        tmp_path.write_text(lines + ("\n" if lines else ""), encoding="utf-8")
        import os
        os.replace(tmp_path, self.log_path)

    # ------------------------------------------------------------------ #
    # Core append                                                          #
    # ------------------------------------------------------------------ #

    async def append_episode(self, ep: dict[str, Any]) -> dict[str, Any]:
        """
        Validate, normalise, and persist a new episode.

        Only episodes with a clear outcome ("success" | "failure" | "neutral")
        are accepted. Chit-chat must be filtered *before* calling this method
        (or the ValueError will surface to the caller).

        Args:
            ep: Raw episode dict. ``id`` and ``ts`` are auto-filled if absent.

        Returns:
            The normalised episode dict that was actually stored.

        Raises:
            ValueError: If ``outcome`` is missing or invalid.
        """
        self._ensure_loaded()
        normalised = _make_episode(ep)
        self._episodes.append(normalised)
        self._append_to_file(normalised)
        return normalised

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
        BaseMemory-compatible wrapper around :meth:`append_episode`.

        Args:
            key:      ``user_id`` for the episode.
            value:    Episode dict (or a summary string for quick saves).
            metadata: Merged into the episode dict; ``outcome`` is required
                      here if `value` is a plain string.

        Raises:
            ValueError: If the resolved episode has no valid ``outcome``.
        """
        if isinstance(value, dict):
            ep = dict(value)
            ep.setdefault("user_id", key)
        else:
            ep = dict(metadata or {})
            ep["user_id"] = key
            ep["summary"] = str(value)

        await self.append_episode(ep)

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        user_id: str | None = None,  # type: ignore[override]
    ) -> list[RetrieveResult]:
        """
        Return the top-k episodes most relevant to *query*.

        Scoring:
          - Keyword overlap between query tokens and the episode corpus
            (task + summary + lesson + tags), normalised by query token count.
          - +0.10 bonus for ``outcome == "success"`` when ``lesson`` is set.

        Args:
            query:   Natural-language query string.
            top_k:   Maximum number of results (default 3).
            user_id: If provided, only episodes for this user are considered.

        Returns:
            List of RetrieveResult dicts ordered by descending score.
            Episodes with score == 0.0 are excluded.
        """
        self._ensure_loaded()

        pool = (
            [ep for ep in self._episodes if ep.get("user_id") == user_id]
            if user_id is not None
            else list(self._episodes)
        )

        query_tokens = _tokenize(query)

        scored: list[tuple[float, dict[str, Any]]] = []
        for ep in pool:
            score = _score_episode(query_tokens, ep)
            if score > 0.0:
                # Clamp here so stored scores are always in [0.0, 1.0]
                scored.append((min(1.0, score), ep))

        # Sort by score descending, then by recency (ts) descending as tiebreaker
        scored.sort(key=lambda x: (x[0], x[1].get("ts", 0.0)), reverse=True)

        results: list[RetrieveResult] = []
        for score, ep in scored[:top_k]:
            content_parts = [f"[{ep['outcome'].upper()}] {ep['task']}"]
            if ep.get("summary"):
                content_parts.append(ep["summary"])
            if ep.get("lesson"):
                content_parts.append(f"Lesson: {ep['lesson']}")

            results.append(
                self._make_result(
                    content=" | ".join(content_parts),
                    score=round(min(1.0, score), 4),
                    source=self.SOURCE,
                    metadata={
                        "id": ep["id"],
                        "user_id": ep["user_id"],
                        "ts": ep["ts"],
                        "outcome": ep["outcome"],
                        "tags": ep.get("tags", []),
                    },
                )
            )
        return results

    async def clear(self, key: str | None = None) -> None:  # type: ignore[override]
        """
        Remove episodes from the log.

        Args:
            key: ``user_id`` whose episodes to delete.
                 If ``None``, wipe the entire log.
        """
        self._ensure_loaded()

        if key is None:
            # Wipe all
            self._episodes = []
            self._rewrite_file([])
        else:
            user_id = str(key)
            kept = [ep for ep in self._episodes if ep.get("user_id") != user_id]
            self._episodes = kept
            self._rewrite_file(kept)

    # ------------------------------------------------------------------ #
    # Extra helpers                                                        #
    # ------------------------------------------------------------------ #

    async def get_all(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """
        Return all episodes, optionally filtered by user_id.

        Args:
            user_id: Filter to this user's episodes. ``None`` → return all.

        Returns:
            List of episode dicts in append order (oldest first).
        """
        self._ensure_loaded()
        if user_id is None:
            return list(self._episodes)
        return [ep for ep in self._episodes if ep.get("user_id") == user_id]

    @property
    def episode_count(self) -> int:
        """Total number of stored episodes (including all users)."""
        self._ensure_loaded()
        return len(self._episodes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"EpisodicMemory(log_path={str(self.log_path)!r}, episodes={self.episode_count})"
