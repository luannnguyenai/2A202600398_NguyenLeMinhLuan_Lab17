"""
memory/semantic.py — Semantic Memory using ChromaDB or fallback keyword search.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memory.base import BaseMemory, RetrieveResult

logger = logging.getLogger(__name__)

try:
    import chromadb
    from langchain_openai import OpenAIEmbeddings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

def _tokenize(text: str) -> set[str]:
    """Simple tokenizer for fallback keyword search."""
    return {
        w.strip(".,;:!?\"'()[]{}").lower() 
        for w in text.split() 
        if len(w.strip(".,;:!?\"'()[]{}")) >= 2
    }

class SemanticMemory(BaseMemory):
    """
    Semantic Memory — stores documents using vector embeddings (ChromaDB) 
    or falls back to keyword matching if dependencies are missing.
    """
    SOURCE = "semantic"

    def __init__(
        self,
        persist_path: str = ".chroma",
        collection: str = "lab17",
        embedding_model: str = "text-embedding-3-small"
    ) -> None:
        self.persist_path = persist_path
        self.collection_name = collection
        self.embedding_model = embedding_model
        
        self._backend = "chroma" if HAS_CHROMA else "keyword"
        self._client = None
        self._collection = None
        self._embeddings = None
        
        self._fallback_docs: list[dict[str, Any]] = []
        self._ready = False

        logger.info(f"[SemanticMemory] Backend: {self._backend}")

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        if self._backend == "chroma":
            if self._client is None:
                self._client = chromadb.PersistentClient(path=self.persist_path)
                self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
        else:
            self._load_fallback_corpus()
            
        self._ready = True

    def _load_fallback_corpus(self) -> None:
        corpus_dir = Path("data/corpus")
        if not corpus_dir.exists():
            return
        
        for p in corpus_dir.glob("*.*"):
            if p.suffix in (".md", ".txt"):
                content = p.read_text(encoding="utf-8")
                self._fallback_docs.append({
                    "id": p.name,
                    "text": content,
                    "source": p.name
                })

    async def ingest(self, docs: list[dict[str, Any]], force: bool = False) -> None:
        """
        Ingest documents. 
        docs format: [{"id": str, "text": str, "source": str}]
        """
        await self._ensure_ready()
        
        if not docs:
            return

        if self._backend == "chroma":
            existing = self._collection.get(include=["metadatas"])
            existing_ids = set(existing["ids"])
            
            to_add = []
            for d in docs:
                if force or d["id"] not in existing_ids:
                    to_add.append(d)
            
            if to_add:
                texts = [d["text"] for d in to_add]
                metadatas = [{"source": d.get("source", "unknown")} for d in to_add]
                ids = [d["id"] for d in to_add]
                
                embeddings = await self._embeddings.aembed_documents(texts)
                
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
        else:
            existing_ids = {d["id"] for d in self._fallback_docs}
            for d in docs:
                if force or d["id"] not in existing_ids:
                    self._fallback_docs.append({
                        "id": d["id"],
                        "text": d["text"],
                        "source": d.get("source", "unknown")
                    })

    async def save(self, key: str, value: Any, metadata: dict | None = None) -> None:
        """BaseMemory compat."""
        source = (metadata or {}).get("source", "unknown")
        await self.ingest([{"id": key, "text": str(value), "source": source}])

    async def retrieve(self, query: str, top_k: int = 4) -> list[RetrieveResult]:
        """
        Retrieve relevant documents.
        Returns: list[{"content": text, "score": 1-distance, "source": meta.source, "metadata": {"doc_id": id}}]
        """
        await self._ensure_ready()
        
        if self._backend == "chroma":
            query_embedding = await self._embeddings.aembed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            ret_results = []
            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]
                ids = results["ids"][0]
                
                for i in range(len(docs)):
                    # dist is typically L2 or cosine distance.
                    # Normalized to [0, 1] loosely for scoring
                    score = max(0.0, 1.0 - (dists[i] / 2.0))
                    ret_results.append(
                        self._make_result(
                            content=docs[i],
                            score=score,
                            source=metas[i].get("source", "unknown"),
                            metadata={"doc_id": ids[i]}
                        )
                    )
            return ret_results
        else:
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []
                
            scored = []
            for doc in self._fallback_docs:
                doc_tokens = _tokenize(doc["text"])
                overlap = len(query_tokens & doc_tokens)
                if overlap > 0:
                    score = min(1.0, overlap / len(query_tokens))
                    scored.append((score, doc))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            ret_results = []
            for score, doc in scored[:top_k]:
                ret_results.append(
                    self._make_result(
                        content=doc["text"],
                        score=score,
                        source=doc.get("source", "unknown"),
                        metadata={"doc_id": doc["id"]}
                    )
                )
            return ret_results

    async def clear(self, key: str | None = None) -> None:
        """Drop collection or clear fallback docs."""
        await self._ensure_ready()
        if self._backend == "chroma":
            if self._client and self.collection_name:
                try:
                    self._client.delete_collection(self.collection_name)
                    self._collection = self._client.get_or_create_collection(name=self.collection_name)
                except Exception:
                    pass
        else:
            self._fallback_docs.clear()
