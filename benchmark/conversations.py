"""
benchmark/conversations.py — Evaluation conversation fixtures.

Each conversation is a list of turns used to test a specific memory
capability. The benchmark runner replays these conversations through the
agent and scores the responses.

Conversation structure:
    {
        "id": str,              # Unique conversation identifier
        "name": str,            # Human-readable test name
        "tags": list[str],      # Capability tags (e.g. ["short_term", "recall"])
        "turns": [
            {
                "user": str,    # User utterance
                "expected_keywords": list[str],  # Keywords expected in response
                "note": str,    # Explanation of what is being tested
            }
        ]
    }
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Conversation 1 — Short-term context recall
# Tests: Does the agent remember user name mentioned two turns ago?
# ---------------------------------------------------------------------------

CONV_SHORT_TERM_RECALL: dict = {
    "id": "stm_recall_001",
    "name": "Short-term name recall",
    "tags": ["short_term", "recall"],
    "turns": [
        {
            "user": "Hi, my name is Alice.",
            "expected_keywords": ["Alice", "hello", "hi"],
            "note": "Agent should acknowledge the name.",
        },
        {
            "user": "What is the capital of France?",
            "expected_keywords": ["Paris"],
            "note": "Factual question — tests that agent still functions.",
        },
        {
            "user": "Do you remember my name?",
            "expected_keywords": ["Alice"],
            "note": "Should recall name from short-term memory.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Conversation 2 — Long-term knowledge retrieval
# Tests: Does the agent use stored facts to answer follow-up questions?
# ---------------------------------------------------------------------------

CONV_LONG_TERM_RETRIEVAL: dict = {
    "id": "ltm_retrieval_001",
    "name": "Long-term knowledge retrieval",
    "tags": ["long_term", "retrieval"],
    "turns": [
        {
            "user": "Please remember that our project deadline is March 31st.",
            "expected_keywords": ["March", "deadline", "remember"],
            "note": "Agent should store this fact in long-term memory.",
        },
        {
            "user": "Tell me about the weather today.",
            "expected_keywords": [],  # Any weather response is acceptable
            "note": "Distractor turn to test persistence of LTM.",
        },
        {
            "user": "When is our project deadline?",
            "expected_keywords": ["March", "31"],
            "note": "Should retrieve deadline from long-term memory.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Conversation 3 — Episodic episode recall
# Tests: Can the agent recall what was discussed in an earlier session?
# ---------------------------------------------------------------------------

CONV_EPISODIC_RECALL: dict = {
    "id": "epi_recall_001",
    "name": "Episodic session recall",
    "tags": ["episodic", "recall"],
    "turns": [
        {
            "user": "Yesterday we talked about neural networks.",
            "expected_keywords": ["neural", "networks"],
            "note": "Agent should log this as an episodic memory.",
        },
        {
            "user": "What did we discuss before?",
            "expected_keywords": ["neural", "networks"],
            "note": "Should recall recent episodic context.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Conversation 4 — Semantic knowledge graph query
# Tests: Does the agent use structured facts to answer relational questions?
# ---------------------------------------------------------------------------

CONV_SEMANTIC_QUERY: dict = {
    "id": "sem_query_001",
    "name": "Semantic knowledge graph query",
    "tags": ["semantic", "graph"],
    "turns": [
        {
            "user": "Alice works at TechCorp as a senior engineer.",
            "expected_keywords": ["Alice", "TechCorp"],
            "note": "Agent should extract and store triples.",
        },
        {
            "user": "Where does Alice work?",
            "expected_keywords": ["TechCorp"],
            "note": "Should answer from semantic graph.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Conversation 5 — Memory conflict resolution
# Tests: When new info contradicts old, does the agent handle correctly?
# ---------------------------------------------------------------------------

CONV_CONFLICT_RESOLUTION: dict = {
    "id": "conflict_001",
    "name": "Memory conflict resolution",
    "tags": ["conflict", "update"],
    "turns": [
        {
            "user": "My favourite colour is blue.",
            "expected_keywords": ["blue"],
            "note": "Stores initial preference.",
        },
        {
            "user": "Actually, my favourite colour is now green.",
            "expected_keywords": ["green"],
            "note": "Updates preference — old value should be superseded.",
        },
        {
            "user": "What is my favourite colour?",
            "expected_keywords": ["green"],
            "note": "Should return updated value, not 'blue'.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Master list of all conversations
# ---------------------------------------------------------------------------

CONVERSATIONS: list[dict] = [
    CONV_SHORT_TERM_RECALL,
    CONV_LONG_TERM_RETRIEVAL,
    CONV_EPISODIC_RECALL,
    CONV_SEMANTIC_QUERY,
    CONV_CONFLICT_RESOLUTION,
]
