import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

from memory import ShortTermMemory, LongTermProfileMemory, EpisodicMemory, SemanticMemory, ContextBudget
from agent.graph import build_graph, invoke_turn

console = Console()

async def amain():
    console.print(Panel("[bold cyan]Lab 17 — Multi-Memory Agent[/bold cyan]\nType 'exit' to quit.", title="Welcome"))
    
    memories = {
        "short_term": ShortTermMemory(),
        "long_term": LongTermProfileMemory(),
        "episodic": EpisodicMemory(),
        "semantic": SemanticMemory(),
        "budget": ContextBudget()
    }
    
    graph = build_graph(memories)
    user_id = "test_user"
    
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
            
        result = await invoke_turn(graph, memories, user_id, user_input)
        
        response = result.get("response", "[No response]")
        retrieved = result.get("retrieved_from", [])
        debug = result.get("debug", {})
        budget_stats = debug.get("budget_stats", {})
        
        console.print(Panel(response, title="[bold green]Agent[/bold green]", border_style="green"))
        
        if retrieved:
            layers = ", ".join(f"[yellow]{l}[/yellow]" for l in retrieved)
            console.print(f"  [dim]🧠 Retrieved from:[/dim] {layers}")
        else:
            console.print("  [dim]🧠 Retrieved from: (no memory used)[/dim]")
            
        if budget_stats:
            console.print(f"  [dim]📊 Budget stats: {budget_stats['total_tokens']} tokens, evicted: {budget_stats['evicted']}[/dim]\n")

def main():
    asyncio.run(amain())

if __name__ == "__main__":
    main()
