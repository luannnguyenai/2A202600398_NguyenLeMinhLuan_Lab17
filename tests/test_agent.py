"""
tests/test_agent.py — Agent tests
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.graph import build_graph, invoke_turn
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermProfileMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.budget import ContextBudget

@pytest.fixture
def memories(tmp_path):
    mem = {
        "short_term": ShortTermMemory(),
        "long_term": LongTermProfileMemory(redis_url=None, json_path=str(tmp_path / "profile.json")),
        "episodic": EpisodicMemory(log_path=str(tmp_path / "ep.jsonl")),
        "semantic": SemanticMemory(),
        "budget": ContextBudget()
    }
    mem["semantic"]._backend = "keyword"
    mem["semantic"]._fallback_docs = []
    mem["semantic"]._ready = False
    return mem

@pytest.fixture
def graph(memories):
    return build_graph(memories)

@pytest.mark.asyncio
@patch("agent.nodes.ChatOpenAI")
@patch("agent.router.ChatOpenAI")
async def test_name_recall(mock_router_llm, mock_node_llm, graph, memories):
    mock_router_instance = MagicMock()
    mock_router_llm.return_value = mock_router_instance
    mock_router_instance.invoke.return_value.content = '{"intent": "preference"}'
    
    mock_node_instance = AsyncMock()
    mock_node_llm.return_value = mock_node_instance
    
    mock_node_instance.ainvoke.side_effect = [
        MagicMock(content="Chào Linh"),
        MagicMock(content='{"facts": [{"key": "name", "value": "Linh"}]}')
    ]
    
    await invoke_turn(graph, memories, "user1", "Tôi tên Linh")
    
    profile = await memories["long_term"].get_profile("user1")
    assert profile.get("name") == "Linh"
    
    mock_router_instance.invoke.return_value.content = '{"intent": "preference"}'
    
    mock_node_instance.ainvoke.side_effect = [
        MagicMock(content="Tên bạn là Linh"),
        MagicMock(content='{"facts": []}')
    ]
    
    res = await invoke_turn(graph, memories, "user1", "Tên tôi là gì?")
    assert "Linh" in res["response"]

@pytest.mark.asyncio
@patch("agent.nodes.ChatOpenAI")
@patch("agent.router.ChatOpenAI")
async def test_episodic_save(mock_router_llm, mock_node_llm, graph, memories):
    mock_router_instance = MagicMock()
    mock_router_llm.return_value = mock_router_instance
    mock_router_instance.invoke.return_value.content = '{"intent": "experience"}'
    
    mock_node_instance = AsyncMock()
    mock_node_llm.return_value = mock_node_instance
    mock_node_instance.ainvoke.return_value.content = "Ghi nhớ thành công"
    
    await invoke_turn(graph, memories, "user1", "Nhớ lần trước tôi đã fix bug này")
    
    eps = await memories["episodic"].get_all("user1")
    assert len(eps) == 1
    assert eps[0]["outcome"] == "success"
    assert eps[0]["task"] == "Nhớ lần trước tôi đã fix bug này"
