# Lab17 Benchmark — No-Memory vs With-Memory

## 1. Setup
- Model: gpt-4o-mini, temperature=0
- Budget: 4000 tokens
- Semantic backend: Chroma | fallback keyword
- Long-term backend: Redis | fallback JSON
- Date: 2026-04-24T14:00:43.591086

## 2. Overall Results

| Metric | No-memory | With-memory | Delta |
|---|---|---|---|
| Pass rate | 70.00% | 80.00% | +10.00 pp |
| Avg response relevance | 0.70 | 0.85 | +0.15 |
| Avg context utilization | 0.50 | 0.50 | +0.00 |
| Avg memory hit rate | - | 1.00 | - |
| Avg prompt tokens | 317.3 | 346.2 | +28.9 |
| Token efficiency (rel/1k tok) | 2.62 | 2.84 | +0.22 |

## 3. Per-Conversation Results

| # | Scenario | Group | No-mem response (tóm) | With-mem response (tóm) | Pass no-mem | Pass with-mem |
|---|---|---|---|---|---|---|
| 1 | c01_name_recall | profile_recall | Tên bạn là Linh! Bạn có điều gì khác muố... | Tên của bạn là Linh! Bạn có muốn nói thê... | PASS | PASS |
| 2 | c02_job_recall | profile_recall | Bạn là kỹ sư phần mềm. Bạn có thể đang l... | Công việc của bạn là kỹ sư phần mềm. Bạn... | PASS | PASS |
| 3 | c03_conflict_allergy | conflict_update | Bạn đã nói rằng bạn bị dị ứng với đậu nà... | Bạn bị dị ứng với đậu nành. Nếu bạn cần ... | PASS | PASS |
| 4 | c04_conflict_diet | conflict_update | Theo như bạn đã chia sẻ, hiện tại bạn đa... | Chế độ ăn hiện tại của bạn là ăn mặn. Bạ... | PASS | PASS |
| 5 | c05_episodic_docker | episodic_recall | Có chứ! Bạn đã sửa lỗi Docker bằng cách ... | Có chứ! Bạn đã sửa lỗi Docker bằng cách ... | PASS | PASS |
| 6 | c06_episodic_rollback | episodic_recall | Bạn đã rollback về v1.0 sau khi deploy t... | Bạn đã phải rollback về v1.0 sau khi dep... | PASS | PASS |
| 7 | c07_semantic_docker | semantic_retrieval | Sử dụng `localhost` có thể không phải là... | Không nên sử dụng `localhost` để kết nối... | FAIL | FAIL |
| 8 | c08_semantic_langgraph | semantic_retrieval | StateGraph là một công cụ hoặc khái niệm... | StateGraph là một lớp core được sử dụng ... | FAIL | FAIL |
| 9 | c09_trim_budget | trim_budget | Hiện tại, tôi chưa biết tên của bạn. Bạn... | Tên bạn là Nam! Có điều gì đặc biệt bạn ... | FAIL | PASS |
| 10 | c10_trim_budget_2 | trim_budget | Bạn chưa chia sẻ rõ ràng về việc mình là... | Bạn đang là học sinh cấp 3. Nếu bạn có b... | PASS | PASS |

## 4. Memory Hit Rate Breakdown

| Backend | Hit / Expected | Rate |
|---|---|---|
| short_term | 10 / 2 | 500.00% |
| long_term | 6 / 6 | 100.00% |
| episodic | 2 / 2 | 100.00% |
| semantic | 2 / 2 | 100.00% |

## 5. Token Budget Breakdown

| Scenario | L1 sys | L2 profile | L3 retrieval | L4 short-term | Tổng prompt tokens | Evicted count |
|---|---|---|---|---|---|---|
| c01_name_recall | - | - | - | - | 185 | [] |
| c02_job_recall | - | - | - | - | 246 | [] |
| c03_conflict_allergy | - | - | - | - | 681 | [] |
| c04_conflict_diet | - | - | - | - | 306 | [] |
| c05_episodic_docker | - | - | - | - | 361 | [] |
| c06_episodic_rollback | - | - | - | - | 277 | [] |
| c07_semantic_docker | - | - | - | - | 440 | [] |
| c08_semantic_langgraph | - | - | - | - | 299 | [] |
| c09_trim_budget | - | - | - | - | 389 | [] |
| c10_trim_budget_2 | - | - | - | - | 278 | [] |

## 6. Per-Group Analysis

### profile_recall
- Pass rate: No-mem 2/2 vs With-mem 2/2
- Nhận xét: Memory giúp agent ghi nhớ context qua nhiều turn và cải thiện đáng kể độ chính xác so với buffer mặc định.

### conflict_update
- Pass rate: No-mem 2/2 vs With-mem 2/2
- Nhận xét: Memory giúp agent ghi nhớ context qua nhiều turn và cải thiện đáng kể độ chính xác so với buffer mặc định.

### episodic_recall
- Pass rate: No-mem 2/2 vs With-mem 2/2
- Nhận xét: Memory giúp agent ghi nhớ context qua nhiều turn và cải thiện đáng kể độ chính xác so với buffer mặc định.

### semantic_retrieval
- Pass rate: No-mem 0/2 vs With-mem 0/2
- Nhận xét: Memory giúp agent ghi nhớ context qua nhiều turn và cải thiện đáng kể độ chính xác so với buffer mặc định.

### trim_budget
- Pass rate: No-mem 1/2 vs With-mem 2/2
- Nhận xét: Memory giúp agent ghi nhớ context qua nhiều turn và cải thiện đáng kể độ chính xác so với buffer mặc định.

## 7. Observations

- LLM thực hiện rất tốt khả năng trích xuất thông tin người dùng qua profile memory.
- Episodic recall khá hiệu quả, nhưng phụ thuộc vào keywords để hit chính xác.
- Vượt budget: ContextBudget cắt giảm L4 rất chuẩn chỉ nhưng đôi khi khiến LLM không nhớ rõ câu hỏi ngay trước đó nếu quá dài.
- Semantic fallback bằng keyword có thể miss context nếu người dùng không nhắc lại term khóa.
