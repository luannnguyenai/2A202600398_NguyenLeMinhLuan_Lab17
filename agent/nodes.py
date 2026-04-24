import json
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.runnables import RunnableConfig

from agent.state import MemoryState
from agent.router import classify_intent
from agent.prompt import build_prompt
from memory.budget import ContextBudget, Chunk, Priority

async def classify_node(state: MemoryState, config: RunnableConfig) -> dict[str, Any]:
    intent = classify_intent(state["user_input"])
    return {"intent": intent}

async def retrieve_memory_node(state: MemoryState, config: RunnableConfig) -> dict[str, Any]:
    intent = state.get("intent", "chitchat")
    memories = config["configurable"]["memories"]
    
    st = memories["short_term"]
    lt = memories["long_term"]
    ep = memories["episodic"]
    sem = memories["semantic"]
    
    user_id = state["user_id"]
    query = state["user_input"]
    
    # Always fetch short term
    st_res = await st.retrieve(query, top_k=None)
    messages = [res["metadata"] for res in st_res]
    
    profile_dict = {}
    episodes_list = []
    semantic_list = []
    retrieved_from = ["short_term"]
    
    if intent == "preference":
        lt_res = await lt.retrieve(query, top_k=5, user_id=user_id)
        for r in lt_res:
            profile_dict[r["metadata"]["fact_key"]] = r["content"].split(": ", 1)[1] if ": " in r["content"] else r["content"]
        retrieved_from.append("long_term")
    elif intent == "experience":
        ep_res = await ep.retrieve(query, top_k=3, user_id=user_id)
        episodes_list = ep_res
        retrieved_from.append("episodic")
    elif intent == "factual":
        sem_res = await sem.retrieve(query, top_k=4)
        semantic_list = sem_res
        retrieved_from.append("semantic")
        
    return {
        "messages": messages,
        "user_profile": profile_dict,
        "episodes": episodes_list,
        "semantic_hits": semantic_list,
        "retrieved_from": retrieved_from
    }

async def pack_context_node(state: MemoryState, config: RunnableConfig) -> dict[str, Any]:
    memories = config["configurable"]["memories"]
    budget: ContextBudget = memories["budget"]
    
    chunks = []
    chunks.append(Chunk(content="SYSTEM", priority=Priority.L1_SYSTEM, tokens=100, score=1.0))
    
    for k, v in state.get("user_profile", {}).items():
        text = f"{k}: {v}"
        chunks.append(Chunk(content=text, priority=Priority.L2_PROFILE, tokens=budget.count_tokens(text), score=1.0, source=k))
        
    for ep in state.get("episodes", []):
        text = ep["content"]
        chunks.append(Chunk(content=text, priority=Priority.L3_RETRIEVAL, tokens=budget.count_tokens(text), score=ep["score"], source=ep["metadata"]["id"]))
        
    for sh in state.get("semantic_hits", []):
        text = sh["content"]
        chunks.append(Chunk(content=text, priority=Priority.L3_RETRIEVAL, tokens=budget.count_tokens(text), score=sh["score"], source=sh["metadata"].get("doc_id", "")))
        
    for msg in state.get("messages", []):
        text = f"{msg['role']}: {msg['content']}"
        chunks.append(Chunk(content=text, priority=Priority.L4_SHORT_TERM, tokens=budget.count_tokens(text), score=1.0, source=text))
        
    packed = budget.pack(chunks)
    stats = budget.last_pack_stats()
    
    new_profile = {}
    new_episodes = []
    new_semantic = []
    new_messages = []
    
    for c in packed:
        if c.priority == Priority.L2_PROFILE:
            new_profile[c.source] = state["user_profile"][c.source]
        elif c.priority == Priority.L3_RETRIEVAL:
            if c.source.startswith("ep_"):
                for ep in state.get("episodes", []):
                    if ep["metadata"]["id"] == c.source:
                        new_episodes.append(ep)
                        break
            else:
                for sh in state.get("semantic_hits", []):
                    if sh["metadata"].get("doc_id") == c.source:
                        new_semantic.append(sh)
                        break
        elif c.priority == Priority.L4_SHORT_TERM:
            for msg in state.get("messages", []):
                if f"{msg['role']}: {msg['content']}" == c.source:
                    new_messages.append(msg)
                    break
                    
    final_messages = [m for m in state.get("messages", []) if m in new_messages]

    debug = state.get("debug", {})
    debug["budget_stats"] = stats

    return {
        "user_profile": new_profile,
        "episodes": new_episodes,
        "semantic_hits": new_semantic,
        "messages": final_messages,
        "debug": debug
    }

async def generate_node(state: MemoryState, config: RunnableConfig) -> dict[str, Any]:
    prompt_msgs = build_prompt(state)
    llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.7)
    
    msgs = []
    for m in prompt_msgs:
        if m["role"] == "system":
            msgs.append(SystemMessage(content=m["content"]))
        else:
            msgs.append(HumanMessage(content=m["content"]))
            
    resp = await llm.ainvoke(msgs)
    return {"response": str(resp.content)}

async def save_memory_node(state: MemoryState, config: RunnableConfig) -> dict[str, Any]:
    memories = config["configurable"]["memories"]
    intent = state.get("intent")
    user_id = state["user_id"]
    user_input = state["user_input"]
    response = state["response"]
    
    st = memories["short_term"]
    await st.save("user", user_input)
    await st.save("assistant", response)
    
    if intent == "preference":
        try:
            llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.0)
            sys_msg = SystemMessage(
                content='''Trích xuất các fact cá nhân dạng {key, value} từ câu user.
Nếu user SỬA/ĐÍNH CHÍNH fact cũ (từ khóa 'À nhầm', 'không phải', 'thực ra'), vẫn trả fact MỚI với cùng key — LongTermProfileMemory sẽ tự overwrite.
Chỉ sử dụng các key sau (snake_case, lowercase): name, age, gender, occupation, location, allergy, diet, language, hobby, preference.
Trả JSON {"facts": [{"key": "...", "value": "..."}]} hoặc {"facts": []}.'''
            )
            hum_msg = HumanMessage(content=user_input)
            ext_resp = await llm.ainvoke([sys_msg, hum_msg], response_format={"type": "json_object"})
            data = json.loads(str(ext_resp.content))
            facts = data.get("facts", [])
            lt = memories["long_term"]
            valid_keys = {"name", "age", "gender", "occupation", "location", "allergy", "diet", "language", "hobby", "preference"}
            for f in facts:
                k = str(f.get("key", "")).lower().strip()
                v = str(f.get("value", "")).strip()
                if k in valid_keys and v:
                    await lt.save(f"{user_id}/{k}", v, metadata={"source": user_input})
        except Exception as e:
            pass
            
    if intent == "experience":
        ep = memories["episodic"]
        await ep.save(
            user_id,
            {
                "task": user_input,
                "outcome": "success",
                "summary": response,
                "lesson": "None",
                "tags": ["experience"]
            }
        )
        
    return {}
