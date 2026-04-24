"""
tests/test_router.py
"""
import pytest
from agent.router import classify_intent

def test_classify_intent_preference():
    assert classify_intent("Tôi thích ăn chay") == "preference"

def test_classify_intent_experience():
    assert classify_intent("Nhớ lần trước bạn đã nói gì không") == "experience"

def test_classify_intent_factual():
    assert classify_intent("Docker là gì") == "factual"

def test_classify_intent_chitchat():
    assert classify_intent("Chào bạn, bạn khỏe không?") == "chitchat"
