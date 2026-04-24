"""
agent/prompt.py — Prompt templates for the Multi-Memory Agent.

All prompt strings are centralised here to make iteration easy without
touching node logic.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a helpful AI assistant with access to multiple memory systems.

When answering, prioritise information retrieved from memory over your
parametric knowledge. Cite your memory sources when relevant.

Current context from memory:
{context}
"""

# ---------------------------------------------------------------------------
# Human message template
# ---------------------------------------------------------------------------

HUMAN_TEMPLATE = "{user_message}"

# ---------------------------------------------------------------------------
# Assembled chat prompt
# ---------------------------------------------------------------------------

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE),
    ]
)

# ---------------------------------------------------------------------------
# Memory routing prompt (used by router to decide which layers to query)
# ---------------------------------------------------------------------------

ROUTING_SYSTEM_TEMPLATE = """\
You are a memory router. Given the user's message, decide which memory
layers should be queried to best answer it.

Available layers: short_term, long_term, episodic, semantic

Return a JSON array of layer names, e.g.: ["short_term", "semantic"]
Only include layers that are likely to contain relevant information.
"""

ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTING_SYSTEM_TEMPLATE),
        ("human", "User message: {user_message}"),
    ]
)
