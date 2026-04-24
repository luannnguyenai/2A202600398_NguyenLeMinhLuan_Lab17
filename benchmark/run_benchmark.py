"""
benchmark/run_benchmark.py
"""
import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmark.conversations import load_conversations
from benchmark.metrics import (
    response_relevance,
    context_utilization,
    token_efficiency,
    memory_hit_rate,
    summarize,
)
from memory import ShortTermMemory, LongTermProfileMemory, EpisodicMemory, SemanticMemory, ContextBudget
from agent.graph import build_graph, invoke_turn

console = Console()

def reset_data_dirs():
    """Wipes out data directory to ensure a clean slate for memory backends."""
    if Path("data").exists():
        shutil.rmtree("data", ignore_errors=True)
    if Path(".chroma").exists():
        shutil.rmtree(".chroma", ignore_errors=True)

async def run_scenario(conversations, mode: str, max_tokens: int):
    results = []
    for conv in conversations:
        reset_data_dirs()
        
        # Ingest semantic if we are using memory
        if mode == "with_memory":
            import benchmark.ingest_mock as im
            await im.ingest_test_corpus()

        memories = {
            "short_term": ShortTermMemory(),
            "long_term": LongTermProfileMemory(),
            "episodic": EpisodicMemory(),
            "semantic": SemanticMemory(),
            "budget": ContextBudget(max_tokens=max_tokens)
        }
        
        graph = build_graph(memories)
        user_id = f"user_{conv['id']}"
        
        final_res = None
        for turn in conv["turns"]:
            # If no-memory, we can wipe short-term before each turn if we want strict no memory,
            # but usually short-term buffer is allowed in no-memory baseline.
            # We'll just not use other memories for no_memory, but we can't easily disable them if graph is fixed.
            # Instead, we can just clear everything for no_memory after every turn, or just mock them.
            # Wait, the instruction says: "dùng 1 graph "naive" chỉ có short-term buffer... hoặc reset mỗi turn".
            if mode == "no_memory":
                await memories["long_term"].clear()
                await memories["episodic"].clear()
                # we let short term stay for conversational flow
            
            res = await invoke_turn(graph, memories, user_id, turn["text"])
            final_res = res
            
        # Calc metrics on final turn
        response = final_res["response"]
        expected = conv.get("expected_contains", [])
        expected_hit = conv.get("expected_memory_hit", [])
        
        rel = response_relevance(response, expected)
        debug = final_res.get("debug", {})
        debug["retrieved_from"] = final_res.get("retrieved_from", [])
        util = context_utilization(debug)
        
        # To get prompt tokens accurately, we can extract from debug if available, else estimate
        stats = debug.get("budget_stats", {})
        prompt_tokens = stats.get("total_tokens", max_tokens)
        
        eff = token_efficiency(prompt_tokens, rel)
        hit_rate = memory_hit_rate(debug["retrieved_from"], expected_hit)
        
        passed = (rel >= 1.0)
        
        results.append({
            "id": conv["id"],
            "group": conv["group"],
            "passed": passed,
            "metrics": {
                "relevance": rel,
                "utilization": util,
                "efficiency": eff,
                "hit_rate": hit_rate
            },
            "response": response
        })
        
    return results

async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--budget", type=int, default=4000)
    args = parser.parse_args()

    os.environ["MODEL"] = args.model
    
    # Write a quick ingest mock for semantic
    Path("benchmark/ingest_mock.py").write_text('''
import asyncio
from pathlib import Path
from memory.semantic import SemanticMemory
async def ingest_test_corpus():
    Path("data/corpus").mkdir(parents=True, exist_ok=True)
    Path("data/corpus/faq_docker.md").write_text("connect container using service name as hostname")
    Path("data/corpus/faq_langgraph.md").write_text("StateGraph is the core class for multi-actor apps")
    mem = SemanticMemory()
    docs = [
        {"id": "doc1", "text": "connect container using service name as hostname", "source": "faq_docker.md"},
        {"id": "doc2", "text": "StateGraph is the core class for multi-actor apps", "source": "faq_langgraph.md"}
    ]
    await mem.ingest(docs)
    ''')

    convs = load_conversations()
    
    console.print("[bold]Running NO-MEMORY baseline...[/bold]")
    no_mem_results = await run_scenario(convs, "no_memory", args.budget)
    
    console.print("[bold]Running WITH-MEMORY agent...[/bold]")
    with_mem_results = await run_scenario(convs, "with_memory", args.budget)
    
    Path("benchmark/results").mkdir(parents=True, exist_ok=True)
    Path("benchmark/results/no_memory.json").write_text(json.dumps(no_mem_results, indent=2))
    Path("benchmark/results/with_memory.json").write_text(json.dumps(with_mem_results, indent=2))
    
    no_mem_sum = summarize(no_mem_results)
    with_mem_sum = summarize(with_mem_results)
    
    table = Table(title="Benchmark Results (No Memory vs With Memory)")
    table.add_column("#")
    table.add_column("Group")
    table.add_column("Scenario")
    table.add_column("No-Mem Pass")
    table.add_column("With-Mem Pass")
    table.add_column("Δ Relevance")
    
    for i, conv in enumerate(convs):
        nm_res = next(r for r in no_mem_results if r["id"] == conv["id"])
        wm_res = next(r for r in with_mem_results if r["id"] == conv["id"])
        
        nm_pass = "✅" if nm_res["passed"] else "❌"
        wm_pass = "✅" if wm_res["passed"] else "❌"
        d_rel = wm_res["metrics"]["relevance"] - nm_res["metrics"]["relevance"]
        
        table.add_row(
            str(i+1),
            conv["group"],
            conv["id"],
            nm_pass,
            wm_pass,
            f"{d_rel:+.2f}"
        )
        
    console.print(table)
    
    # Generate BENCHMARK.md
    md_content = f"""# Multi-Memory Agent Benchmark Report

## Overall Metrics
| Metric | No-Memory | With-Memory |
|--------|-----------|-------------|
| Pass Rate | {no_mem_sum['overall']['pass_rate']:.2%} | {with_mem_sum['overall']['pass_rate']:.2%} |
| Relevance | {no_mem_sum['overall']['relevance']:.2f} | {with_mem_sum['overall']['relevance']:.2f} |
| Hit Rate | {no_mem_sum['overall']['hit_rate']:.2f} | {with_mem_sum['overall']['hit_rate']:.2f} |
| Efficiency | {no_mem_sum['overall']['efficiency']:.2f} | {with_mem_sum['overall']['efficiency']:.2f} |

## Scenario Details
"""
    for i, conv in enumerate(convs):
        nm_res = next(r for r in no_mem_results if r["id"] == conv["id"])
        wm_res = next(r for r in with_mem_results if r["id"] == conv["id"])
        md_content += f"- **{conv['id']}** ({conv['group']}): No-Mem [{'PASS' if nm_res['passed'] else 'FAIL'}], With-Mem [{'PASS' if wm_res['passed'] else 'FAIL'}]\n"

    Path("BENCHMARK.md").write_text(md_content)
    console.print("\n[bold green]Report saved to BENCHMARK.md[/bold green]")

if __name__ == "__main__":
    asyncio.run(amain())
