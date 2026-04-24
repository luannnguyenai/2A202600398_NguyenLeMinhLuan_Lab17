
import asyncio
from pathlib import Path
from memory.semantic import SemanticMemory
async def ingest_test_corpus():
    Path("data/corpus").mkdir(parents=True, exist_ok=True)
    Path("data/corpus/faq_docker.md").write_text("sử dụng service name thay vì localhost làm hostname")
    Path("data/corpus/faq_langgraph.md").write_text("StateGraph là core class cho các ứng dụng multi-actor")
    mem = SemanticMemory()
    docs = [
        {"id": "doc1", "text": "kết nối container bằng cách dùng service name làm hostname", "source": "faq_docker.md"},
        {"id": "doc2", "text": "StateGraph là core class cho các ứng dụng multi-actor", "source": "faq_langgraph.md"}
    ]
    await mem.ingest(docs)
    