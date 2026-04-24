"""
memory/semantic.py — Semantic Memory as an in-memory knowledge graph.

Semantic memory stores structured facts as (subject, predicate, object)
triples in a directed graph. It supports querying by subject or object
and reasoning over chains of relationships.

This implementation uses a plain Python dict-of-dicts adjacency list.
For production use, consider replacing with NetworkX or a graph database.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from memory.base import BaseMemory


# Type alias for a triple
Triple = tuple[str, str, str]  # (subject, predicate, object)


class SemanticMemory(BaseMemory):
    """
    Semantic Memory layer using an in-memory directed knowledge graph.

    Graph structure:
        _graph[subject][predicate] = list[object]

    Attributes:
        _graph: Adjacency dict storing (subject → predicate → [object]).
        _triples: Flat list of all stored triples for iteration.
    """

    def __init__(self) -> None:
        """Initialise an empty knowledge graph."""
        self._graph: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._triples: list[Triple] = []

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        **kwargs: Any,
    ) -> None:
        """
        Add a (subject, predicate, object) triple to the knowledge graph.

        Args:
            subject:   The subject entity (e.g. "Alice").
            predicate: The relationship (e.g. "works_at").
            obj:       The object entity (e.g. "OpenAI").

        TODO:
            - Avoid duplicate triples.
            - Optionally normalise strings (lowercase, strip).
        """
        # TODO: implement deduplication
        self._graph[subject][predicate].append(obj)
        self._triples.append((subject, predicate, obj))

    def search(self, query: str, k: int = 5, **kwargs: Any) -> list[str]:
        """
        Find triples where the subject or object matches the query.

        Args:
            query: Entity name to search for (exact or substring match).
            k:     Max number of results.

        Returns:
            List of formatted triple strings, e.g. ["Alice works_at OpenAI"].

        TODO:
            - Implement substring / fuzzy matching.
            - Support multi-hop reasoning (optional).
        """
        # TODO: implement
        results = []
        for subj, pred, obj in self._triples:
            if query.lower() in subj.lower() or query.lower() in obj.lower():
                results.append(f"{subj} {pred} {obj}")
                if len(results) >= k:
                    break
        return results

    def query(self, subject: str) -> list[str]:
        """
        Return all facts about a given subject entity.

        Args:
            subject: The entity to look up.

        Returns:
            List of strings like "predicate: object".

        TODO:
            - Look up self._graph[subject] and format output.
        """
        # TODO: implement proper formatting
        if subject not in self._graph:
            return []
        facts = []
        for pred, objects in self._graph[subject].items():
            for obj in objects:
                facts.append(f"{pred}: {obj}")
        return facts

    def clear(self, **kwargs: Any) -> None:
        """
        Clear all triples from the knowledge graph.

        TODO:
            - Reset self._graph and self._triples.
        """
        self._graph.clear()
        self._triples.clear()

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------

    def load_from_file(self, filepath: str) -> None:
        """
        Load triples from a TSV or JSON file into the graph.

        Expected TSV format: subject\\tpredicate\\tobject (one triple per line).

        Args:
            filepath: Path to the input file.

        TODO:
            - Detect format (tsv / json) by file extension.
            - Parse and call self.add() for each triple.
        """
        # TODO: implement
        pass

    @property
    def triple_count(self) -> int:
        """Return the total number of stored triples."""
        return len(self._triples)
