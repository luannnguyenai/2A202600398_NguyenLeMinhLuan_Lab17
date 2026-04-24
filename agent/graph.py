"""
agent/graph.py — LangGraph graph definition for the Multi-Memory Agent.

Graph topology:
  START → budget_node → retrieve_node → generate_node → store_node → END

Each node is a plain Python function defined in agent/nodes.py.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import budget_node, retrieve_node, generate_node, store_node


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph agent graph.

    Node wiring:
      1. budget_node   — compute available token budget for memory context
      2. retrieve_node — query memory layers and assemble context
      3. generate_node — call LLM with context + user message
      4. store_node    — persist exchange to memory layers

    Returns:
        A compiled LangGraph StateGraph ready to be invoked with an
        initial AgentState dict.

    TODO:
        - Add conditional edges for memory routing (optional optimisation).
        - Add error-handling / retry edges if generate_node fails.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("budget_node", budget_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("store_node", store_node)

    # Wire edges: linear pipeline
    builder.add_edge(START, "budget_node")
    builder.add_edge("budget_node", "retrieve_node")
    builder.add_edge("retrieve_node", "generate_node")
    builder.add_edge("generate_node", "store_node")
    builder.add_edge("store_node", END)

    return builder.compile()
