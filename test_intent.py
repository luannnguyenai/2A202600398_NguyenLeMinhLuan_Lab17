import asyncio
from agent.router import classify_intent
from benchmark.conversations import load_conversations

async def main():
    for c in load_conversations():
        for t in c["turns"]:
            intent = classify_intent(t["text"])
            print(f"{t['text'][:30]:<30} -> {intent}")

asyncio.run(main())
