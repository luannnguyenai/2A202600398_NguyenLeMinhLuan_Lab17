# REFLECTION.md — Lab 17 Multi-Memory Agent

Trong quá trình thực hiện Lab 17: Build Multi-Memory Agent, tôi đã rút ra được nhiều bài học quan trọng về kiến trúc quản lý ngữ cảnh, tính hiệu quả cũng như các rủi ro bảo mật tiềm ẩn khi tích hợp các bộ nhớ dài hạn vào hệ thống LLM. Dưới đây là phân tích chi tiết của tôi.

## 1. Memory nào giúp agent nhất? Memory nào rủi ro nhất?

Dựa vào các kết quả giả định từ báo cáo `BENCHMARK.md`, bộ nhớ mang lại sự cải thiện rõ rệt nhất cho Agent chính là **Long-Term Profile Memory**. Việc có khả năng thu thập, định danh và truy xuất thông tin người dùng ngay lập tức giúp tỷ lệ chính xác (Pass rate) của nhóm test `profile_recall` và `conflict_update` tăng mạnh (thường là +100pp) so với mức cơ sở (baseline) NO-MEMORY. Người dùng không cần nhắc lại tên hay công việc của mình ở các lượt hội thoại sau, tạo ra một luồng tương tác cực kỳ tự nhiên.

Tuy nhiên, đây cũng chính là **loại bộ nhớ rủi ro nhất**. Profile memory lưu trữ trực tiếp các dữ liệu cá nhân nhạy cảm (PII) như tên gọi, và đặc biệt là tình trạng sức khoẻ (ví dụ: dị ứng đậu nành, dị ứng sữa bò, chế độ ăn). Dữ liệu sức khỏe (Health Data) thuộc diện dữ liệu cực kỳ nhạy cảm theo các tiêu chuẩn như GDPR/HIPAA. Việc lưu giữ vĩnh viễn trong một backend mà không được bảo vệ mã hoá hợp lý có thể gây rủi ro vô cùng lớn.

## 2. Rủi ro privacy/PII cụ thể

Trong hệ thống kiến trúc hiện tại, có hai rủi ro bảo mật chính có thể xảy ra:
- **(a) Lưu trữ PII không được mã hoá:** Dữ liệu profile (bao gồm cả dữ liệu sức khoẻ) hiện đang được lưu thẳng dưới dạng văn bản thuần (plain text) trong `data/profile.json` (hoặc trong Redis). Điều này dẫn tới nguy cơ lộ lọt nếu máy chủ bị xâm nhập.
- **(b) Rò rỉ qua vết Episodic Log:** Episodic memory append toàn bộ dữ kiện trải nghiệm, điều này đôi khi có thể vô tình chứa luôn các prompt nguyên gốc của user kèm theo thông tin nhạy cảm.

Bên cạnh đó, **rủi ro Retrieval sai lệch** ở layer Semantic là cực kỳ nghiêm trọng: Nếu bộ máy trả về một chunk văn bản không đúng phiên, hoặc truy vấn nhầm lẫn dữ kiện của `user_id` này sang `user_id` khác (ví dụ quên filter user metadata), toàn bộ hệ thống sẽ rò rỉ (leak) dữ liệu giữa các cá nhân khác nhau.

## 3. TTL / Deletion / Consent

Để giải quyết các rủi ro riêng tư trên, tôi đề xuất thiết kế quyền kiểm soát như sau:
- **Thời gian sống (TTL):** 
  - `short_term` chỉ nên tồn tại trong phạm vi 1 session, bị xóa ngay sau khi tắt trình duyệt.
  - `episodic` có thể giữ trong 30 ngày (30d) để Agent rút kinh nghiệm các lỗi gần nhất mà không phình to quá mức.
  - `profile` sẽ là *user-controlled* (người dùng tự quyết định thời điểm huỷ bỏ).
- **Deletion API:** Chức năng xoá dữ liệu đã được tôi thiết kế và implement sẵn thông qua phương thức `LongTermProfileMemory.delete_fact` và `EpisodicMemory.clear(user_id)`. Cần công khai các lệnh gọi này trên giao diện.
- **Quyền đồng thuận (Consent):** Trước khi Agent kích hoạt pipeline để trích xuất `save_memory_node` vào profile, cần hiện một prompt xin quyền (Consent) xác nhận. Hoặc cung cấp một cờ `opt-out` rõ ràng trong tuỳ chọn State để user từ chối việc bị thu thập sở thích.

## 4. Limitation kỹ thuật của solution hiện tại

Về mặt công nghệ, giải pháp đa bộ nhớ hiện tại vẫn đang tồn tại một số hạn chế (Limitation):
1. **Classifier định tuyến chưa hoàn hảo:** Router intent dùng LLM có tỉ lệ lỗi và độ trễ phản hồi. Nếu router phân loại nhầm (như báo cáo ở benchmark, intent có thể miss), bộ nhớ sai sẽ được kích hoạt, dẫn tới việc context cần thiết bị thiếu.
2. **Conflict Handling bị giới hạn từ vựng:** Logic hiện tại ở LongTermProfileMemory phụ thuộc vào schema key và overwrite `same-key`. Nhưng nếu user phủ định cách phát biểu, ví dụ: "Thực ra tôi không còn dị ứng nữa", mô hình có thể không capture được "negation" để xoá bỏ key, mà lại lưu thành một fact sai lệch.
3. **Phụ thuộc Embedding Model:** Semantic Memory đòi hỏi sử dụng OpenAI API liên tục cho mọi document ingestion và truy vấn, dẫn đến chi phí (cost) và độ trễ (latency) không hề nhỏ.
4. **Thiếu vắng Memory Consolidation:** Do hệ thống Episodic lưu theo dạng append log liên tục, khối lượng JSONL sẽ lớn dần lên và gây cản trở việc search. Thiếu đi một quy trình đúc kết/tổng hợp (consolidation) định kỳ sẽ khiến hệ thống "scale poorly" khi hoạt động lâu dài.
