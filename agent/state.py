"""
agent/state.py
"""
from typing import TypedDict, Annotated, Any

class MemoryState(TypedDict):
    user_id: str
    user_input: str
    messages: list[dict[str, Any]]
    user_profile: dict[str, str]
    episodes: list[dict[str, Any]]
    semantic_hits: list[dict[str, Any]]
    memory_budget: int
    intent: str
    retrieved_from: list[str]
    response: str
    debug: dict[str, Any]
