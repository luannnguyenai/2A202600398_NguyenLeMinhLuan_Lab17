# LangGraph FAQ

**Q: What is a StateGraph?**
A: A `StateGraph` is the core class in LangGraph used to build stateful, multi-actor applications. You define a graph by adding nodes (functions that update the state) and edges (which specify the order of execution). 

**Q: How do nodes work?**
A: Each node in the graph takes the current state as input and returns an update to the state. The state is typically defined as a `TypedDict`. LangGraph automatically manages the passing of state between nodes.
