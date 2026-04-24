"""
main.py — CLI Chat Loop for Lab17 Multi-Memory Agent

Usage:
    python main.py

The loop:
  1. Reads user input from stdin.
  2. Invokes the LangGraph graph via agent.graph.build_graph().
  3. Prints the agent response.
  4. Prints debug info: state.retrieved_from (which memory layers were hit).
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Load .env before importing agent modules that read env vars
load_dotenv()

from agent.graph import build_graph  # noqa: E402

console = Console()


def print_welcome() -> None:
    """Print a welcome banner to the console."""
    console.print(
        Panel(
            "[bold cyan]Lab 17 — Multi-Memory Agent[/bold cyan]\n"
            "Type your message and press Enter. Type [bold red]exit[/bold red] to quit.",
            title="Welcome",
            border_style="cyan",
        )
    )


def print_response(response: str, retrieved_from: list[str]) -> None:
    """
    Pretty-print the agent response and memory provenance info.

    Args:
        response: The agent's text response.
        retrieved_from: List of memory layer names that contributed context
                        (e.g. ["short_term", "episodic"]).
    """
    console.print(Panel(response, title="[bold green]Agent[/bold green]", border_style="green"))

    if retrieved_from:
        layers = ", ".join(f"[yellow]{l}[/yellow]" for l in retrieved_from)
        console.print(f"  [dim]🧠 Retrieved from:[/dim] {layers}\n")
    else:
        console.print("  [dim]🧠 Retrieved from: (no memory used)[/dim]\n")


def main() -> None:
    """
    Entry point for the CLI chat loop.

    TODO:
        - Build the LangGraph graph via build_graph().
        - Maintain a session_id across turns.
        - Pass session_id + user_message as initial state to graph.invoke().
        - Extract response text and state.retrieved_from from the result.
    """
    print_welcome()

    graph = build_graph()
    session_id = "cli-session-001"
    turn = 0

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        turn += 1

        # TODO: build initial state dict and call graph.invoke()
        initial_state = {
            "session_id": session_id,
            "turn": turn,
            "user_message": user_input,
            "response": "",
            "retrieved_from": [],
        }

        try:
            result = graph.invoke(initial_state)
            response = result.get("response", "[No response]")
            retrieved_from = result.get("retrieved_from", [])
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            continue

        print_response(response, retrieved_from)


if __name__ == "__main__":
    main()
