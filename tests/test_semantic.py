"""
tests/test_semantic.py — Unit tests for SemanticMemory.
"""
from __future__ import annotations

import pytest

from memory.semantic import SemanticMemory

@pytest.fixture
def semantic_mem():
    # Initialize without chromadb to ensure fallback keyword matching is tested
    mem = SemanticMemory()
    mem._backend = "keyword"
    mem._fallback_docs = []
    mem._ready = False
    return mem

@pytest.mark.asyncio
async def test_semantic_retrieve_fallback(semantic_mem: SemanticMemory):
    # Ingest the corpus
    await semantic_mem.ingest([
        {
            "id": "faq_docker.md_0",
            "text": "When using Docker Compose, you can connect to another container using its service name as the hostname. Ensure both containers are on the same Docker network.",
            "source": "faq_docker.md"
        },
        {
            "id": "faq_langgraph.md_0",
            "text": "A StateGraph is the core class in LangGraph used to build stateful, multi-actor applications.",
            "source": "faq_langgraph.md"
        },
        {
            "id": "faq_python.md_0",
            "text": "async and await are Python keywords used to define and execute asynchronous code.",
            "source": "faq_python.md"
        }
    ])
    
    # Query for docker network
    results = await semantic_mem.retrieve("docker network", top_k=4)
    
    assert len(results) > 0, "Retrieve should return results"
    
    # The top result should be from docker faq and have score > 0
    top_result = results[0]
    assert top_result["source"] == "faq_docker.md"
    assert top_result["score"] > 0.0

    # Ensure shape is correct
    assert "content" in top_result
    assert "score" in top_result
    assert "source" in top_result
    assert "metadata" in top_result
    assert "doc_id" in top_result["metadata"]
