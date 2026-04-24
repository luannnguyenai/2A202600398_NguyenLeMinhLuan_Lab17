"""
memory/long_term.py — Long-Term Memory backed by ChromaDB.

Long-term memory persists factual knowledge using vector embeddings.
Documents are embedded with OpenAI Embeddings and stored in a persistent
ChromaDB collection on disk.

Storage schema:
    Collection: "long_term_memory"
    Documents:  Raw text chunks
    Metadata:   {"session_id": str, "turn": int, "timestamp": str}
"""

from __future__ import annotations

import os
from typing import Any

from memory.base import BaseMemory


class LongTermMemory(BaseMemory):
    """
    Long-Term Memory layer using ChromaDB as the vector store.

    Attributes:
        chroma_path: Filesystem path to the ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        _client: Lazy-initialised ChromaDB PersistentClient.
        _collection: Lazy-initialised ChromaDB Collection.
        _embeddings: Lazy-initialised OpenAIEmbeddings instance.
    """

    COLLECTION_NAME = "long_term_memory"

    def __init__(
        self,
        chroma_path: str | None = None,
    ) -> None:
        """
        Initialise LongTermMemory.

        Args:
            chroma_path: Path to ChromaDB storage. Defaults to CHROMA_PATH env var.
        """
        self.chroma_path = chroma_path or os.getenv("CHROMA_PATH", ".chroma")
        self._client = None
        self._collection = None
        self._embeddings = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self):
        """
        Return (and lazily create) the ChromaDB collection.

        TODO:
            - Import chromadb and langchain_openai.OpenAIEmbeddings.
            - Create PersistentClient(path=self.chroma_path).
            - Get or create collection with embedding_function.
        """
        if self._collection is None:
            # TODO: initialise client and collection
            pass
        return self._collection

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> None:
        """
        Embed and store a text document in ChromaDB.

        Args:
            text:     The text content to store.
            metadata: Optional metadata dict (session_id, turn, etc.).
            doc_id:   Optional unique document ID (auto-generated if None).

        TODO:
            - Generate embedding for `text`.
            - collection.add(documents=[text], embeddings=[...], metadatas=[metadata], ids=[doc_id]).
        """
        # TODO: implement
        pass

    def search(self, query: str, k: int = 5, **kwargs: Any) -> list[str]:
        """
        Perform semantic similarity search over stored documents.

        Args:
            query: Natural language query string.
            k:     Number of results to return.

        Returns:
            List of top-k document strings ordered by similarity.

        TODO:
            - Embed query with self._embeddings.
            - collection.query(query_embeddings=[...], n_results=k).
            - Return result["documents"][0].
        """
        # TODO: implement
        return []

    def clear(self, **kwargs: Any) -> None:
        """
        Delete all documents in the collection.

        TODO:
            - collection.delete(where={}) or recreate collection.
        """
        # TODO: implement
        pass
