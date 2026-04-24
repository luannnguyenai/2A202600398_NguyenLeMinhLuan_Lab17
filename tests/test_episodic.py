"""
tests/test_episodic.py — Unit tests for EpisodicMemory.

Run with:
    pytest tests/test_episodic.py -v

Tests:
    1. test_keyword_retrieval_top1     — 3 episodes, query returns correct top-1
    2. test_user_id_filter             — retrieve only sees episodes for given user
    3. test_invalid_outcome_rejected   — chit-chat without outcome raises ValueError
    4. test_success_bonus_applied      — success+lesson earns +0.1 bonus score
    5. test_clear_user                 — clear(user_id) removes only that user's episodes
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from memory.episodic import EpisodicMemory, _tokenize, _score_episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_file(tmp_path: Path) -> str:
    """Return a fresh temporary JSONL path per test."""
    return str(tmp_path / "test_episodic_log.jsonl")


@pytest.fixture
def em(log_file: str) -> EpisodicMemory:
    """Fresh EpisodicMemory backed by a temp file."""
    return EpisodicMemory(log_path=log_file)


# ---------------------------------------------------------------------------
# Episode factories
# ---------------------------------------------------------------------------

def _ep(task: str, summary: str, outcome: str = "success",
        lesson: str | None = None, tags: list[str] | None = None,
        user_id: str = "alice") -> dict:
    return {
        "user_id": user_id,
        "task": task,
        "outcome": outcome,
        "summary": summary,
        "lesson": lesson,
        "tags": tags or [],
    }


# ---------------------------------------------------------------------------
# Test 1 — keyword retrieval returns the correct top-1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keyword_retrieval_top1(em: EpisodicMemory):
    """
    Given 3 episodes with distinct topics, a keyword query must return
    the most relevant episode as the first result.
    """
    await em.append_episode(_ep(
        task="Deploy FastAPI service to AWS Lambda",
        summary="Successfully deployed the API with cold-start under 300ms.",
        outcome="success",
        lesson="Use provisioned concurrency to avoid Lambda cold starts.",
        tags=["aws", "lambda", "fastapi", "deploy"],
    ))
    await em.append_episode(_ep(
        task="Fix Redis connection timeout in production",
        summary="Redis client was not pooling connections properly.",
        outcome="failure",
        lesson="Always configure connection pool size explicitly.",
        tags=["redis", "production", "timeout"],
    ))
    await em.append_episode(_ep(
        task="Write unit tests for payment processing module",
        summary="Achieved 95% coverage on the payment module.",
        outcome="success",
        lesson="Mock external payment gateway calls to keep tests fast.",
        tags=["testing", "pytest", "payment"],
    ))

    assert em.episode_count == 3

    # Query targeting the Redis episode
    results = await em.retrieve("Redis connection pool timeout fix", top_k=3)

    assert len(results) >= 1, "Should return at least 1 result for a specific query"

    top1 = results[0]

    # Top result must be the Redis episode
    assert "redis" in top1["content"].lower() or "redis" in str(top1["metadata"]["tags"]).lower(), (
        f"Top-1 result should be the Redis episode, got: {top1['content']}"
    )
    assert top1["source"] == "episodic"
    assert 0.0 < top1["score"] <= 1.0

    # Results must be in descending score order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by descending score"

    # RetrieveResult shape validation
    for r in results:
        assert "content" in r
        assert "score" in r
        assert "source" in r
        assert "metadata" in r
        assert "id" in r["metadata"]
        assert "outcome" in r["metadata"]


# ---------------------------------------------------------------------------
# Test 2 — user_id filter isolates episodes by user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_id_filter(em: EpisodicMemory):
    """
    retrieve(..., user_id="bob") must only surface episodes belonging to bob,
    even when alice has episodes with higher keyword relevance.
    """
    # Alice's episodes — highly relevant to "database migration"
    await em.append_episode(_ep(
        task="Run database migration on PostgreSQL",
        summary="Migrated schema from v1 to v2 using Alembic.",
        outcome="success",
        lesson="Always back up database before running migrations.",
        tags=["database", "postgresql", "migration", "alembic"],
        user_id="alice",
    ))
    await em.append_episode(_ep(
        task="Optimise database query for large tables",
        summary="Added composite index; query time dropped from 5s to 50ms.",
        outcome="success",
        tags=["database", "optimisation", "index"],
        user_id="alice",
    ))

    # Bob's episode — less relevant to "database migration" but it's Bob's only one
    await em.append_episode(_ep(
        task="Set up CI/CD pipeline with GitHub Actions",
        summary="Configured automated tests and deployment pipeline.",
        outcome="success",
        tags=["ci", "github", "devops"],
        user_id="bob",
    ))

    # Query without filter — alice's episodes should dominate
    all_results = await em.retrieve("database migration", top_k=3)
    user_ids_all = [r["metadata"]["user_id"] for r in all_results]
    assert "alice" in user_ids_all, "Alice's episodes should appear without filter"

    # Query filtered to bob — must return ONLY bob's episode(s)
    bob_results = await em.retrieve("database migration", top_k=3, user_id="bob")
    for r in bob_results:
        assert r["metadata"]["user_id"] == "bob", (
            f"All results must belong to 'bob', got {r['metadata']['user_id']}"
        )

    # Query filtered to alice — must return ONLY alice's episode(s)
    alice_results = await em.retrieve("database", top_k=5, user_id="alice")
    assert len(alice_results) == 2, f"Alice has 2 database episodes, got {len(alice_results)}"
    for r in alice_results:
        assert r["metadata"]["user_id"] == "alice"


# ---------------------------------------------------------------------------
# Test 3 — invalid outcome raises ValueError (chit-chat guard)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_outcome_rejected(em: EpisodicMemory):
    """
    Episodes without a valid outcome must be rejected with ValueError.
    This prevents chit-chat from polluting the episodic store.
    """
    with pytest.raises(ValueError, match="outcome"):
        await em.append_episode({
            "user_id": "alice",
            "task": "Casual conversation about the weather",
            "outcome": "chat",       # invalid
            "summary": "We talked about rain.",
        })

    with pytest.raises(ValueError, match="outcome"):
        await em.append_episode({
            "user_id": "alice",
            "task": "Missing outcome entirely",
            # no outcome key
            "summary": "Something happened.",
        })

    # Store must remain empty after rejections
    assert em.episode_count == 0, "No episode should be stored after rejected saves"


# ---------------------------------------------------------------------------
# Test 4 — success + lesson bonus increases score
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_success_bonus_applied(em: EpisodicMemory):
    """
    An episode with outcome=success AND lesson set must have a raw score
    exactly 0.10 higher than the same episode without a lesson.

    _score_episode returns an unclamped value so the bonus is always visible
    even when the base score is already 1.0. retrieve() clamps the final
    score to [0.0, 1.0].
    """
    from memory.episodic import _score_episode, _tokenize

    # Use a query that partially matches so base_score < 1.0,
    # making the +0.10 bonus unambiguous
    query_tokens = _tokenize("cache eviction strategy")

    ep_with_lesson = {
        "task": "improve cache eviction",
        "outcome": "success",
        "summary": "Switched to LRU and hit rate improved.",
        "lesson": "LRU eviction is best for hot-key workloads.",
        "tags": ["cache"],
    }
    ep_without_lesson = {
        "task": "improve cache eviction",
        "outcome": "success",
        "summary": "Switched to LRU and hit rate improved.",
        "lesson": None,
        "tags": ["cache"],
    }
    ep_failure_with_lesson = {
        "task": "improve cache eviction",
        "outcome": "failure",
        "summary": "Switched to LRU and hit rate improved.",
        "lesson": "LRU eviction is best for hot-key workloads.",
        "tags": ["cache"],
    }

    score_with = _score_episode(query_tokens, ep_with_lesson)
    score_without = _score_episode(query_tokens, ep_without_lesson)
    score_failure = _score_episode(query_tokens, ep_failure_with_lesson)

    # _score_episode returns RAW (unclamped) scores — bonus is always +0.10
    assert score_with > score_without, (
        f"success+lesson ({score_with:.3f}) should beat success+no-lesson ({score_without:.3f})"
    )
    assert abs(score_with - score_without) == pytest.approx(0.10, abs=1e-6), (
        f"Bonus should be exactly 0.10, got diff={score_with - score_without:.4f}"
    )
    # failure+lesson must NOT earn the bonus
    assert score_failure == score_without, (
        f"failure+lesson ({score_failure:.3f}) should equal no-lesson ({score_without:.3f})"
    )

    # Sanity: retrieve() must clamp returned scores to ≤ 1.0
    await em.append_episode({**ep_with_lesson, "user_id": "test"})
    results = await em.retrieve("cache eviction strategy", top_k=1, user_id="test")
    assert results, "Should return the stored episode"
    assert results[0]["score"] <= 1.0, "retrieve() scores must be clamped to ≤ 1.0"


# ---------------------------------------------------------------------------
# Test 5 — clear(user_id) removes only that user's episodes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_user(em: EpisodicMemory):
    """
    clear(key=user_id) must remove only episodes for the given user;
    episodes for other users must be untouched.
    """
    await em.append_episode(_ep("Task A", "Summary A", user_id="alice"))
    await em.append_episode(_ep("Task B", "Summary B", user_id="alice"))
    await em.append_episode(_ep("Task C", "Summary C", user_id="bob"))

    assert em.episode_count == 3

    # Clear only alice
    await em.clear(key="alice")

    assert em.episode_count == 1, (
        f"Only bob's episode should remain, got count={em.episode_count}"
    )

    remaining = await em.get_all()
    assert all(ep["user_id"] == "bob" for ep in remaining), (
        "Only bob's episode should remain after clearing alice"
    )

    # Verify persisted to disk — reload from file
    em2 = EpisodicMemory(log_path=str(em.log_path))
    all_ep = await em2.get_all()
    assert len(all_ep) == 1
    assert all_ep[0]["user_id"] == "bob"

    # clear(None) wipes everything
    await em.clear()
    assert em.episode_count == 0

    em3 = EpisodicMemory(log_path=str(em.log_path))
    assert (await em3.get_all()) == []
