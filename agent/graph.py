from typing import Any
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.state import MemoryState
from agent.nodes import (
    classify_node,
    retrieve_memory_node,
    pack_context_node,
    generate_node,
    save_memory_node
)

def build_graph(memories: dict[str, Any]):
    builder = StateGraph(MemoryState)
    
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_memory_node)
    builder.add_node("pack", pack_context_node)
    builder.add_node("generate", generate_node)
    builder.add_node("save", save_memory_node)
    
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "pack")
    builder.add_edge("pack", "generate")
    builder.add_edge("generate", "save")
    builder.add_edge("save", END)
    
    return builder.compile()

async def invoke_turn(graph, memories: dict[str, Any], user_id: str, text: str):
    state = {
        "user_id": user_id,
        "user_input": text,
        "messages": [],
        "user_profile": {},
        "episodes": [],
        "semantic_hits": [],
        "memory_budget": memories["budget"].max_tokens,
        "intent": "",
        "retrieved_from": [],
        "response": "",
        "debug": {}
    }
    
    config = RunnableConfig(configurable={"memories": memories})
    return await graph.ainvoke(state, config=config)
