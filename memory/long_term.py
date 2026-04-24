"""
memory/long_term.py — Long-Term Profile Memory with Redis / JSON fallback.

Stores structured user-profile facts under a two-level key:
    profile[user_id][fact_key] → FactEntry

Conflict resolution: **last-write-wins** — a new value always overwrites
the current one, while the previous value is appended to `history` for audit.

Backend selection (resolved once at __init__ time):
  1. Redis  — if `redis_url` is provided AND the connection is healthy.
  2. JSON   — atomic file-based store (write to .tmp then os.replace).

The class is fully async so it slots directly into the LangGraph pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from memory.base import BaseMemory, RetrieveResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal data shapes
# ---------------------------------------------------------------------------

class FactEntry:
    """Single profile fact with conflict-audit history."""

    __slots__ = ("value", "updated_at", "source", "history")

    def __init__(
        self,
        value: str,
        updated_at: float,
        source: str,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.value: str = value
        self.updated_at: float = updated_at
        self.source: str = source
        self.history: list[dict[str, Any]] = history or []

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "updated_at": self.updated_at,
            "source": self.source,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FactEntry":
        return cls(
            value=d["value"],
            updated_at=d.get("updated_at", 0.0),
            source=d.get("source", ""),
            history=list(d.get("history", [])),
        )


# ---------------------------------------------------------------------------
# Redis backend (lazy import — only used when Redis is available)
# ---------------------------------------------------------------------------

_REDIS_KEY_PREFIX = "ltm:profile:"


async def _try_build_redis(redis_url: str):
    """
    Attempt to create and PING a redis.asyncio.Redis connection.

    Returns the client if healthy, or None on any error.
    """
    try:
        import redis.asyncio as aioredis  # noqa: PLC0415

        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        return client
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis connection failed (%s), falling back to JSON.", exc)
        return None


# ---------------------------------------------------------------------------
# LongTermProfileMemory
# ---------------------------------------------------------------------------

class LongTermProfileMemory(BaseMemory):
    """
    Long-Term Profile Memory — persists structured user facts across sessions.

    Storage backend is chosen automatically at construction time:
      • Redis  — async, keyed as ``ltm:profile:<user_id>`` (JSON-serialised dict)
      • JSON   — atomic file write to ``json_path`` (``data/profile.json``)

    Attributes:
        redis_url:   Redis connection URL (``None`` → skip Redis attempt).
        json_path:   Path to the fallback JSON file.
        _redis:      Live ``redis.asyncio.Redis`` client, or ``None``.
        _backend:    Human-readable backend name ("redis" | "json").
        _cache:      In-process profile dict (always kept in sync with backend).
    """

    SOURCE = "long_term_profile"

    def __init__(
        self,
        redis_url: str | None = None,
        json_path: str = "data/profile.json",
    ) -> None:
        """
        Initialise LongTermProfileMemory.

        Backend selection happens lazily in :meth:`_ensure_ready` on the
        first async call so that the constructor stays synchronous (required
        by LangGraph node singletons).

        Args:
            redis_url: Redis URL, e.g. ``"redis://localhost:6379/0"``.
                       If ``None`` the JSON backend is used immediately.
            json_path: Path to the JSON profile file (created if absent).
        """
        self.redis_url: str | None = redis_url or os.getenv("REDIS_URL")
        self.json_path: Path = Path(json_path)
        self._redis = None          # set by _ensure_ready()
        self._backend: str = ""     # "redis" | "json"
        self._ready: bool = False
        # In-process cache — avoids redundant serialisation
        self._cache: dict[str, dict[str, FactEntry]] = {}

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    async def _ensure_ready(self) -> None:
        """
        Resolve backend (once) and load existing data into cache.

        Idempotent — safe to call before every public method.
        """
        if self._ready:
            return

        # --- Try Redis ---
        if self.redis_url:
            self._redis = await _try_build_redis(self.redis_url)

        if self._redis is not None:
            self._backend = "redis"
            await self._load_from_redis()
        else:
            self._backend = "json"
            self._load_from_json()

        logger.info(
            "[LongTermProfileMemory] Backend: %s%s",
            self._backend,
            f" ({self.json_path})" if self._backend == "json" else "",
        )
        self._ready = True

    # ------------------------------------------------------------------ #
    # Backend I/O — Redis                                                  #
    # ------------------------------------------------------------------ #

    async def _load_from_redis(self) -> None:
        """Load all profile keys from Redis into the in-process cache."""
        assert self._redis is not None
        try:
            keys = await self._redis.keys(f"{_REDIS_KEY_PREFIX}*")
            for key in keys:
                user_id = key[len(_REDIS_KEY_PREFIX):]
                raw = await self._redis.get(key)
                if raw:
                    raw_dict: dict[str, Any] = json.loads(raw)
                    self._cache[user_id] = {
                        fk: FactEntry.from_dict(fv)
                        for fk, fv in raw_dict.items()
                    }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load from Redis: %s", exc)

    async def _persist_user_redis(self, user_id: str) -> None:
        """Serialise and store one user's profile to Redis."""
        assert self._redis is not None
        key = f"{_REDIS_KEY_PREFIX}{user_id}"
        payload = {
            fk: fe.to_dict()
            for fk, fe in self._cache.get(user_id, {}).items()
        }
        await self._redis.set(key, json.dumps(payload))

    async def _delete_user_redis(self, user_id: str) -> None:
        assert self._redis is not None
        await self._redis.delete(f"{_REDIS_KEY_PREFIX}{user_id}")

    # ------------------------------------------------------------------ #
    # Backend I/O — JSON                                                   #
    # ------------------------------------------------------------------ #

    def _load_from_json(self) -> None:
        """Load the JSON file into the in-process cache (creates file if absent)."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.json_path.exists():
            self._cache = {}
            return
        try:
            raw = json.loads(self.json_path.read_text(encoding="utf-8"))
            self._cache = {
                user_id: {
                    fk: FactEntry.from_dict(fv)
                    for fk, fv in user_facts.items()
                }
                for user_id, user_facts in raw.items()
            }
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupted profile JSON (%s); starting fresh.", exc)
            self._cache = {}

    def _persist_json(self) -> None:
        """
        Atomically write the full cache to disk.

        Writes to ``<json_path>.tmp`` first, then ``os.replace()`` — ensures
        the file is never half-written even on crash.
        """
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            user_id: {fk: fe.to_dict() for fk, fe in facts.items()}
            for user_id, facts in self._cache.items()
        }
        tmp_path = self.json_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.json_path)

    # ------------------------------------------------------------------ #
    # Shared persist dispatcher                                            #
    # ------------------------------------------------------------------ #

    async def _persist(self, user_id: str | None = None) -> None:
        """
        Persist data to the active backend.

        Args:
            user_id: If provided (Redis only), persist just this user's
                     profile. For JSON the whole file is always rewritten.
        """
        if self._backend == "redis" and self._redis is not None:
            if user_id is not None:
                await _try_persist_redis(self._redis, user_id, self._cache)
            # full-cache redis flush: iterate all dirty users (not needed here)
        else:
            self._persist_json()

    # ------------------------------------------------------------------ #
    # Core fact operations                                                 #
    # ------------------------------------------------------------------ #

    async def save_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        source: str = "",
    ) -> None:
        """
        Upsert a fact for a user.

        Conflict rule — **last-write-wins**:
          • If `key` already exists with the same `value` → no-op.
          • If `key` already exists with a *different* `value` → push old
            ``{value, ts}`` to ``history``, overwrite current value.
          • If `key` is new → create a fresh FactEntry with empty history.

        Args:
            user_id: Logical user identifier.
            key:     Fact key (e.g. ``"allergy"``).
            value:   New fact value (e.g. ``"đậu nành"``).
            source:  The original utterance or document that asserted this fact.
        """
        await self._ensure_ready()

        now = time.time()
        user_facts = self._cache.setdefault(user_id, {})

        if key in user_facts:
            existing = user_facts[key]
            if existing.value == value:
                return  # identical — no-op

            # Conflict: push old value to history, overwrite
            existing.history.append({"value": existing.value, "ts": existing.updated_at})
            existing.value = value
            existing.updated_at = now
            existing.source = source
        else:
            # New fact
            user_facts[key] = FactEntry(
                value=value,
                updated_at=now,
                source=source,
                history=[],
            )

        await self._persist_user(user_id)

    async def get_profile(self, user_id: str) -> dict[str, str]:
        """
        Return a flat ``{fact_key: current_value}`` dict for a user.

        Args:
            user_id: Logical user identifier.

        Returns:
            Dict of current fact values; empty dict if user is unknown.
        """
        await self._ensure_ready()
        return {
            fk: fe.value
            for fk, fe in self._cache.get(user_id, {}).items()
        }

    async def delete_fact(self, user_id: str, key: str) -> None:
        """
        Remove a single fact from a user's profile.

        Args:
            user_id: Logical user identifier.
            key:     Fact key to remove.
        """
        await self._ensure_ready()
        user_facts = self._cache.get(user_id, {})
        if key in user_facts:
            del user_facts[key]
            await self._persist_user(user_id)

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
        BaseMemory-compatible wrapper around :meth:`save_fact`.

        Args:
            key:      ``"<user_id>/<fact_key>"`` (slash-delimited).
            value:    Fact value string.
            metadata: Optional dict; ``source`` key used if present.

        Raises:
            ValueError: If `key` does not contain exactly one ``/``.
        """
        parts = str(key).split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"LongTermProfileMemory.save() key must be '<user_id>/<fact_key>', got '{key}'"
            )
        user_id, fact_key = parts
        source = (metadata or {}).get("source", str(value))
        await self.save_fact(user_id, fact_key, str(value), source=source)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "default",
    ) -> list[RetrieveResult]:
        """
        Return all profile facts for *user_id* as a list of RetrieveResult dicts.

        No semantic ranking is applied — every fact is returned with
        ``score=1.0`` (profile facts are always equally "relevant").
        `top_k` caps the result list for interface consistency.

        Args:
            query:   Ignored (profile lookup is key-based, not semantic).
            top_k:   Maximum results to return.
            user_id: Profile to look up (defaults to ``"default"``).

        Returns:
            List of RetrieveResult dicts, one per stored fact.
        """
        await self._ensure_ready()
        facts = self._cache.get(user_id, {})
        results: list[RetrieveResult] = []
        for fk, fe in list(facts.items())[:top_k]:
            results.append(
                self._make_result(
                    content=f"{fk}: {fe.value}",
                    score=1.0,
                    source=self.SOURCE,
                    metadata={
                        "user_id": user_id,
                        "fact_key": fk,
                        "updated_at": fe.updated_at,
                        "source_utterance": fe.source,
                        "history_length": len(fe.history),
                    },
                )
            )
        return results

    async def clear(self, key: str | None = None) -> None:
        """
        Clear profile data.

        Args:
            key: ``user_id`` to clear. If ``None``, wipe *all* profiles.
        """
        await self._ensure_ready()

        if key is not None:
            user_id = str(key)
            if user_id in self._cache:
                del self._cache[user_id]
            if self._backend == "redis" and self._redis is not None:
                await self._delete_user_redis(user_id)
            else:
                self._persist_json()
        else:
            # Wipe everything
            if self._backend == "redis" and self._redis is not None:
                for uid in list(self._cache.keys()):
                    await self._delete_user_redis(uid)
            self._cache.clear()
            if self._backend == "json":
                self._persist_json()

    # ------------------------------------------------------------------ #
    # Internal persist dispatcher (DRY helper)                             #
    # ------------------------------------------------------------------ #

    async def _persist_user(self, user_id: str) -> None:
        """Persist one user's profile to the active backend."""
        if self._backend == "redis" and self._redis is not None:
            await self._persist_user_redis(user_id)
        else:
            self._persist_json()

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LongTermProfileMemory(backend={self._backend!r}, "
            f"users={list(self._cache.keys())})"
        )


# ---------------------------------------------------------------------------
# Module-level helper (avoids a forward-reference inside the class body)
# ---------------------------------------------------------------------------

async def _try_persist_redis(client, user_id: str, cache: dict) -> None:
    """Serialise and write one user's profile to Redis."""
    try:
        key = f"{_REDIS_KEY_PREFIX}{user_id}"
        payload = {
            fk: fe.to_dict()
            for fk, fe in cache.get(user_id, {}).items()
        }
        await client.set(key, json.dumps(payload))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis persist failed for user %r: %s", user_id, exc)
