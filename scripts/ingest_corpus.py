"""
scripts/ingest_corpus.py — Scripts to ingest corpus files into SemanticMemory.
"""
import asyncio
from pathlib import Path

from memory.semantic import SemanticMemory


def chunk_text(text: str, token_size: int = 500) -> list[str]:
    """Simple chunking by word count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), token_size):
        chunks.append(" ".join(words[i:i + token_size]))
    return chunks

async def main():
    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists():
        print(f"Directory {corpus_dir} does not exist.")
        return

    mem = SemanticMemory()
    docs = []

    for p in corpus_dir.glob("*.*"):
        if p.suffix in (".md", ".txt"):
            content = p.read_text(encoding="utf-8")
            chunks = chunk_text(content, 500)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{p.name}_{i}",
                    "text": chunk,
                    "source": p.name
                })
    
    if docs:
        print(f"Ingesting {len(docs)} chunks...")
        await mem.ingest(docs)
        print("Done.")
    else:
        print("No documents found in data/corpus/")

if __name__ == "__main__":
    asyncio.run(main())
