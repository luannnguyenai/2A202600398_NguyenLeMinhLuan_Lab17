"""
benchmark/conversations.py — Multi-turn conversation tests.
"""
from typing import Any

CONVERSATIONS: list[dict[str, Any]] = [
    {
        "id": "c01_name_recall",
        "group": "profile_recall",
        "turns": [
            {"role": "user", "text": "Tôi tên Linh."},
            {"role": "user", "text": "Hôm nay trời đẹp nhỉ."},
            {"role": "user", "text": "Nhân tiện, tên tôi là gì?"},
        ],
        "expected_contains": ["Linh"],
        "expected_memory_hit": ["long_term"],
    },
    {
        "id": "c02_job_recall",
        "group": "profile_recall",
        "turns": [
            {"role": "user", "text": "Tôi làm nghề kỹ sư phần mềm."},
            {"role": "user", "text": "Bạn thấy nghề này có khó không?"},
            {"role": "user", "text": "Vậy công việc của tôi là gì?"},
        ],
        "expected_contains": ["kỹ sư", "phần mềm"],
        "expected_memory_hit": ["long_term"],
    },
    {
        "id": "c03_conflict_allergy",
        "group": "conflict_update",
        "turns": [
            {"role": "user", "text": "Tôi bị dị ứng sữa bò."},
            {"role": "user", "text": "Bạn có công thức làm bánh nào ngon không?"},
            {"role": "user", "text": "À nhầm, thực ra tôi bị dị ứng đậu nành chứ không phải sữa bò."},
            {"role": "user", "text": "Vậy tóm lại tôi bị dị ứng với gì?"},
        ],
        "expected_contains": ["đậu nành"],
        "expected_memory_hit": ["long_term"],
    },
    {
        "id": "c04_conflict_diet",
        "group": "conflict_update",
        "turns": [
            {"role": "user", "text": "Tôi ăn chay."},
            {"role": "user", "text": "Súp lơ xào có ngon không?"},
            {"role": "user", "text": "Thực ra tôi không ăn chay nữa, tôi chuyển sang ăn mặn rồi."},
            {"role": "user", "text": "Chế độ ăn hiện tại của tôi là gì?"},
        ],
        "expected_contains": ["mặn"],
        "expected_memory_hit": ["long_term"],
    },
    {
        "id": "c05_episodic_docker",
        "group": "episodic_recall",
        "turns": [
            {"role": "user", "text": "Tôi đã sửa lỗi Docker bằng cách dùng host.docker.internal."},
            {"role": "user", "text": "Cái này dùng trên Mac là chuẩn bài."},
            {"role": "user", "text": "Nhớ lần trước tôi đã sửa lỗi Docker thế nào không?"},
        ],
        "expected_contains": ["host.docker.internal"],
        "expected_memory_hit": ["episodic"],
    },
    {
        "id": "c06_episodic_rollback",
        "group": "episodic_recall",
        "turns": [
            {"role": "user", "text": "Hôm trước deploy thất bại nên tôi đã phải rollback về v1.0."},
            {"role": "user", "text": "Mệt mỏi thật sự luôn á."},
            {"role": "user", "text": "Hôm trước deploy lỗi tôi đã xử lý ra sao?"},
        ],
        "expected_contains": ["rollback", "v1.0"],
        "expected_memory_hit": ["episodic"],
    },
    {
        "id": "c07_semantic_docker",
        "group": "semantic_retrieval",
        "turns": [
            {"role": "user", "text": "Chào bạn."},
            {"role": "user", "text": "Bạn có biết làm sao để kết nối container từ container khác không?"},
            {"role": "user", "text": "Tại sao không nên dùng localhost nhỉ?"},
        ],
        "expected_contains": ["service name", "hostname"],
        "expected_memory_hit": ["semantic"],
    },
    {
        "id": "c08_semantic_langgraph",
        "group": "semantic_retrieval",
        "turns": [
            {"role": "user", "text": "Chào bạn, nay tìm hiểu về LangGraph."},
            {"role": "user", "text": "Công cụ này hay phết."},
            {"role": "user", "text": "StateGraph là gì vậy?"},
        ],
        "expected_contains": ["core", "class"],
        "expected_memory_hit": ["semantic"],
    },
    {
        "id": "c09_trim_budget",
        "group": "trim_budget",
        "turns": [
            {"role": "user", "text": "Tôi tên là Nam."},
            {"role": "user", "text": "Một."},
            {"role": "user", "text": "Hai."},
            {"role": "user", "text": "Ba."},
            {"role": "user", "text": "Bốn."},
            {"role": "user", "text": "Năm."},
            {"role": "user", "text": "Sáu."},
            {"role": "user", "text": "Bảy."},
            {"role": "user", "text": "Tám."},
            {"role": "user", "text": "Tên tôi là gì?"},
        ],
        "expected_contains": ["Nam"],
        "expected_memory_hit": ["long_term", "short_term"],
    },
    {
        "id": "c10_trim_budget_2",
        "group": "trim_budget",
        "turns": [
            {"role": "user", "text": "Tôi là học sinh cấp 3."},
            {"role": "user", "text": "Hôm nay tôi ăn cơm."},
            {"role": "user", "text": "Xong tôi đi ngủ."},
            {"role": "user", "text": "Rồi tôi lại thức dậy."},
            {"role": "user", "text": "Đi học bài."},
            {"role": "user", "text": "Đi chơi game."},
            {"role": "user", "text": "Nghe nhạc một chút."},
            {"role": "user", "text": "Tôi đang là học sinh hay sinh viên?"},
        ],
        "expected_contains": ["học sinh"],
        "expected_memory_hit": ["long_term", "short_term"],
    }
]

def load_conversations() -> list[dict[str, Any]]:
    """Return the list of benchmark conversations."""
    return CONVERSATIONS
