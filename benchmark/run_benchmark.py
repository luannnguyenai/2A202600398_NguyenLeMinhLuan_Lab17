"""
benchmark/run_benchmark.py — CLI runner for the benchmark evaluation suite.

Usage:
    python benchmark/run_benchmark.py [--output results/benchmark_results.json]

Steps:
  1. Load all conversations from benchmark/conversations.py.
  2. Build the LangGraph agent graph.
  3. Replay each conversation turn-by-turn, measuring latency.
  4. Score responses with benchmark/metrics.py.
  5. Save results to JSON and print a Rich summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from agent.graph import build_graph  # noqa: E402
from benchmark.conversations import CONVERSATIONS  # noqa: E402
from benchmark.metrics import (  # noqa: E402
    ConversationResult,
    TurnResult,
    compute_metrics,
    score_turn,
)

console = Console()

DEFAULT_OUTPUT = Path(__file__).parent / "results" / "benchmark_results.json"


def run_conversation(
    graph,
    conversation: dict,
    session_prefix: str = "bench",
) -> ConversationResult:
    """
    Replay a single conversation through the agent and collect results.

    Args:
        graph:          Compiled LangGraph graph.
        conversation:   Conversation fixture dict from conversations.py.
        session_prefix: Prefix for auto-generated session IDs.

    Returns:
        ConversationResult with per-turn scores.

    TODO:
        - Maintain session_id across turns (same session).
        - Build initial_state for each turn.
        - Measure wall-clock latency around graph.invoke().
        - Call score_turn() and package into TurnResult.
    """
    session_id = f"{session_prefix}-{conversation['id']}"
    result = ConversationResult(
        conversation_id=conversation["id"],
        conversation_name=conversation["name"],
        tags=conversation["tags"],
    )

    for i, turn in enumerate(conversation["turns"]):
        initial_state = {
            "session_id": session_id,
            "turn": i + 1,
            "user_message": turn["user"],
            "response": "",
            "context": "",
            "retrieved_from": [],
            "memory_budget": 0,
            "metadata": {},
        }

        t0 = time.perf_counter()
        try:
            output = graph.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            output = {"response": f"[ERROR] {exc}", "retrieved_from": []}
        latency = time.perf_counter() - t0

        response = output.get("response", "")
        retrieved_from = output.get("retrieved_from", [])
        hits, misses = score_turn(response, turn.get("expected_keywords", []))

        result.turn_results.append(
            TurnResult(
                turn_index=i,
                user_message=turn["user"],
                response=response,
                expected_keywords=turn.get("expected_keywords", []),
                retrieved_from=retrieved_from,
                latency_seconds=latency,
                keyword_hits=hits,
                keyword_misses=misses,
            )
        )

    return result


def print_summary_table(results: list[ConversationResult], metrics: dict) -> None:
    """
    Print a Rich table summarising benchmark results.

    Args:
        results: List of ConversationResult objects.
        metrics: Aggregate metrics dict from compute_metrics().
    """
    table = Table(title="📊 Benchmark Results", show_lines=True)
    table.add_column("Conversation", style="cyan", no_wrap=True)
    table.add_column("Tags", style="dim")
    table.add_column("Keyword Hit Rate", justify="right")
    table.add_column("Exact Match Rate", justify="right")
    table.add_column("Mean Latency (s)", justify="right")

    for conv in results:
        table.add_row(
            conv.conversation_name,
            ", ".join(conv.tags),
            f"{conv.mean_keyword_hit_rate:.0%}",
            f"{conv.exact_match_rate:.0%}",
            f"{conv.mean_latency:.2f}s",
        )

    console.print(table)
    console.print(
        f"\n[bold]Overall:[/bold] "
        f"KHR={metrics.get('overall_keyword_hit_rate', 0):.0%}  "
        f"EMR={metrics.get('overall_exact_match_rate', 0):.0%}  "
        f"Latency={metrics.get('mean_latency_seconds', 0):.2f}s\n"
    )


def main(output_path: Path = DEFAULT_OUTPUT) -> None:
    """
    Main entry point for the benchmark runner.

    TODO:
        - Build graph.
        - Run all conversations.
        - Compute and print metrics.
        - Save JSON results.
    """
    console.print("[bold cyan]Starting benchmark...[/bold cyan]")
    graph = build_graph()

    all_results: list[ConversationResult] = []
    for conv in CONVERSATIONS:
        console.print(f"  Running: [yellow]{conv['name']}[/yellow]")
        result = run_conversation(graph, conv)
        all_results.append(result)

    metrics = compute_metrics(all_results)
    print_summary_table(all_results, metrics)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "conversations": [
            {
                "id": r.conversation_id,
                "name": r.conversation_name,
                "tags": r.tags,
                "mean_keyword_hit_rate": r.mean_keyword_hit_rate,
                "exact_match_rate": r.exact_match_rate,
                "mean_latency": r.mean_latency,
            }
            for r in all_results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Lab17 benchmark suite.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save JSON results.",
    )
    args = parser.parse_args()
    main(output_path=args.output)
