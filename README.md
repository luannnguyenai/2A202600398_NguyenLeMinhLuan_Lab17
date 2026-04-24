# Lab 17 — Multi-Memory Agent with LangGraph

## Mục tiêu

Xây dựng một AI Agent sử dụng **4 tầng bộ nhớ** (Short-Term, Long-Term, Episodic, Semantic) được điều phối bởi **LangGraph**. Agent có khả năng:

- Nhớ ngữ cảnh hội thoại ngắn hạn (Short-Term Memory — Redis)
- Lưu trữ kiến thức lâu dài (Long-Term Memory — ChromaDB vector store)
- Ghi lại lịch sử các episode (Episodic Memory — JSONL log)
- Suy luận trên tri thức có cấu trúc (Semantic Memory — in-memory graph)
- Tự động quản lý ngân sách token (Memory Budget Manager)

## Cấu trúc thư mục

```
.
├── agent/          # LangGraph graph definition, nodes, router, prompts
├── memory/         # Memory layer implementations
├── benchmark/      # Evaluation suite
├── tests/          # Unit tests
├── data/corpus/    # Knowledge corpus files
├── main.py         # CLI chat loop
└── requirements.txt
```

## Cách chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Cấu hình environment
cp .env.example .env
# Mở .env và điền OPENAI_API_KEY

# 3. Khởi động Redis (Docker)
docker run -d --name lab17-redis -p 6379:6379 redis:7-alpine

# 4. Chạy chat loop
python main.py

# 5. Chạy benchmark đánh giá
python benchmark/run_benchmark.py
```

## Các thành phần chính

| Layer | Storage | Class |
|-------|---------|-------|
| Short-Term | Redis | `ShortTermMemory` |
| Long-Term | ChromaDB | `LongTermMemory` |
| Episodic | JSONL file | `EpisodicMemory` |
| Semantic | In-memory graph | `SemanticMemory` |

## Yêu cầu

- Python 3.11+
- Docker (để chạy Redis)
- OpenAI API Key
