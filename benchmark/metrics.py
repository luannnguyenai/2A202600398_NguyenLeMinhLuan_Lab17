"""
benchmark/metrics.py — Benchmark metrics for multi-memory agent evaluation.
"""
from typing import Any

def response_relevance(response: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    lower_res = response.lower()
    matches = sum(1 for kw in expected if kw.lower() in lower_res)
    return matches / len(expected)

def context_utilization(state_debug: dict[str, Any]) -> float:
    # Estimate from retrieved_from or budget_stats
    budget_stats = state_debug.get("budget_stats", {})
    # It is harder to get exact non-empty memory layers from state_debug if they aren't explicitly tracked,
    # but the instructions say "# memory type được inject non-empty / 4".
    # We can pass retrieved_from in state_debug, or just check the length of retrieved_from.
    # retrieved_from is passed alongside debug, let's assume it's passed as an argument or in debug.
    retrieved = state_debug.get("retrieved_from", [])
    if isinstance(retrieved, list):
        return len(set(retrieved)) / 4.0
    return 0.0

def token_efficiency(prompt_tokens: int, response_relevance: float) -> float:
    return response_relevance / max(prompt_tokens, 1) * 1000.0

def memory_hit_rate(retrieved_from: list[str], expected_hit: list[str]) -> float:
    if not expected_hit:
        return 1.0
    s_ret = set(retrieved_from)
    s_exp = set(expected_hit)
    return len(s_ret & s_exp) / len(s_exp)

def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        g = r.get("group", "unknown")
        groups.setdefault(g, []).append(r)
        
    def avg(lst: list[float]) -> float:
        return sum(lst) / max(len(lst), 1)

    summary = {
        "overall": {
            "relevance": avg([r["metrics"]["relevance"] for r in results]),
            "utilization": avg([r["metrics"]["utilization"] for r in results]),
            "efficiency": avg([r["metrics"]["efficiency"] for r in results]),
            "hit_rate": avg([r["metrics"]["hit_rate"] for r in results]),
            "pass_rate": avg([1.0 if r["passed"] else 0.0 for r in results]),
        },
        "groups": {}
    }
    
    for g, grecs in groups.items():
        summary["groups"][g] = {
            "relevance": avg([r["metrics"]["relevance"] for r in grecs]),
            "utilization": avg([r["metrics"]["utilization"] for r in grecs]),
            "efficiency": avg([r["metrics"]["efficiency"] for r in grecs]),
            "hit_rate": avg([r["metrics"]["hit_rate"] for r in grecs]),
            "pass_rate": avg([1.0 if r["passed"] else 0.0 for r in grecs]),
        }
        
    return summary
