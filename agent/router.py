"""
agent/router.py — Intent classification
"""
import hashlib
import json
import logging
import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_INTENT_CACHE: dict[str, str] = {}

def classify_intent(user_input: str) -> Literal["preference", "factual", "experience", "chitchat"]:
    h = hashlib.md5(user_input.encode("utf-8")).hexdigest()
    if h in _INTENT_CACHE:
        return _INTENT_CACHE[h]

    intent = None
    try:
        llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.0)
        sys_msg = SystemMessage(
            content='You are an intent classifier. Classify the user input into exactly one of:\n'
                    '- "preference": questions or statements about the user\'s name, personal details, likes/dislikes, diet, allergy, or occupation.\n'
                    '- "factual": general knowledge questions, definitions, or asking how things work (e.g. Docker, LangGraph).\n'
                    '- "experience": statements or questions about past actions, past errors, lessons learned, or debugging history.\n'
                    '- "chitchat": general conversational filler, greetings, or off-topic chat.\n'
                    'Output JSON matching schema: {"intent": "...", "reason": "..."}'
        )
        hum_msg = HumanMessage(content=user_input)
        
        resp = llm.invoke([sys_msg, hum_msg], response_format={"type": "json_object"})
        data = json.loads(resp.content)
        intent = data.get("intent", "").lower()
    except Exception as e:
        logger.warning(f"LLM classify failed: {e}. Falling back to rule-based.")

    if intent not in ("preference", "factual", "experience", "chitchat"):
        lower_input = user_input.lower()
        if any(kw in lower_input for kw in ["tôi thích", "dị ứng", "tên tôi", "tôi tên"]):
            intent = "preference"
        elif any(kw in lower_input for kw in ["nhớ lần", "hôm trước", "bạn đã", "đã làm"]):
            intent = "experience"
        elif any(kw in lower_input for kw in ["làm sao", "là gì", "tại sao"]):
            intent = "factual"
        else:
            intent = "chitchat"

    _INTENT_CACHE[h] = intent
    return intent
