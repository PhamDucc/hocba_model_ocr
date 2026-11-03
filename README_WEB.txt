WEB DEMO HỌC BẠ — VINTERN GỐC, CHỈ CPU

CÀI TRONG DỰ ÁN ĐANG CÓ
1. Dừng app.py cũ bằng Ctrl+C. Sao lưu app.py và templates/index.html cũ.
2. Chép các file trong gói này vào thư mục hocba_ocr, cùng cấp .venv_vintern.
   Gói không chứa dữ liệu học sinh, model hoặc môi trường Python.
3. Cài phần web bằng môi trường Vintern đã chạy được:
   .\.venv_vintern\Scripts\python.exe -m pip install -r requirements_web.txt
4. Chạy:
   .\.venv_vintern\Scripts\python.exe app.py
5. Mở http://127.0.0.1:5000
6. Nếu muốn chia sẻ demo, ở terminal khác:
   ngrok http 5000
   Gửi link HTTPS ngrok cho người xem. Giữ cả hai terminal mở.

CÁCH DEMO
- Tải ảnh nguyên trang, chọn đúng bố cục user_002 hoặc user_003.
  Tên file tùy ý: bố cục lấy từ lựa chọn, không suy ra từ tên file.
- Bấm Xem vùng cắt. Xem cả 4 vùng, bảo đảm đủ chữ và dấu.
- Nếu sai vùng, chọn lại bố cục/ảnh và tạo bản xem trước mới; không cố đọc ảnh sai crop.
- Bấm Đọc 4 nhóm. Có tiến độ theo nhóm. Lần đầu có thể tải model.
- Sau khi đọc, đối chiếu OCR gốc với crop, sửa giá trị và tick những trường đã xác nhận.
- Tick ô có giá trị rỗng = xác nhận trường thực sự trống. Nếu khó đọc/chưa chắc, không tick.
- Bấm Lưu các trường đã xác nhận; thao tác này thay toàn bộ danh sách xác nhận hiện có.
  Bỏ tick trường cũ rồi lưu sẽ rút xác nhận trường đó.
- Tải riêng JSON kết quả, OCR gốc, xác nhận và thống kê đánh giá.
- Sửa thủ công không fine-tune model. Chỉ chấm trường đã xác nhận, trước khi sửa.

THỜI GIAN / GIỚI HẠN
- Các lần thử thực tế của bạn khoảng 1–2 phút/4 nhóm CPU; không cam kết mọi máy/ảnh giống vậy.
- Vintern được giữ trong RAM sau lượt đầu; mỗi lúc chỉ 1 ảnh được OCR.
- Dùng lại tiền xử lý 6 ô tối đa + thumbnail và prompt đọc nhóm đã thử.
- Web chỉ dùng vùng cắt cố định của hai bố cục đã kiểm tra, KHÔNG tự nhận diện bố cục mới.
- Chưa đọc bảng quá trình học tập. Chữ bị che/mờ vẫn có thể sai hoặc thiếu.
- Không dùng Paddle hoặc VietOCR trên luồng web mới. Các script cũ của dự án vẫn có thể giữ để đối chiếu.
- App lắng nghe localhost, debug tắt; ngrok chuyển tiếp vào localhost.
- Trang demo không có tài khoản đăng nhập. Giữ link trong nhóm demo, tắt ngrok sau buổi dùng.
- Kết quả lưu trong web_uploads_vintern/<id>/, gồm original.png, 4 crop,
  4 văn bản model, source.json, student_raw.json, student.json, verified_fields.json
  (sau khi lưu xác nhận), evaluation.json/csv và timing.json.
- Hai ảnh cùng tên vẫn tạo hai id riêng, không dùng chung xác nhận.
- Khởi động lại web sẽ mất danh sách tác vụ trong RAM; dữ liệu trên đĩa vẫn còn.
  Tải JSON trước khi tắt nếu cần xem lại; chưa có trang lịch sử hồ sơ.
- Không đóng web khi đang đọc. Lỗi model được ghi trong process.log của lượt chạy và terminal.
- source của file tải xuống là tên ảnh tải lên; toàn bộ dữ liệu gắn với id riêng.

CẢNH BÁO THƯ VIỆN
- Dịch vụ web dùng context limit lấy từ cấu hình language model nếu có, thay vì tự đặt 1700.
- pad token chỉ được cấu hình khi tokenizer cung cấp token id hợp lệ; không đoán id.
- Không cài FlashAttention cho CPU.

KIỂM TRA ĐÃ LÀM KHI ĐÓNG GÓI
- Kiểm tra cú pháp và luồng Flask bằng dữ liệu mô phỏng: upload, xem crop, tiến độ,
  xác nhận/rút xác nhận, tải file, giữ OCR gốc, chặn ảnh/tập trường không hợp lệ.
- Template HTML và cú pháp JavaScript đã kiểm tra; chưa kiểm tra tương tác bằng trình duyệt thật do môi trường không tải được Chromium.
- Không tải hoặc chạy model thật trong môi trường đóng gói. Cần thử lại một lượt trên
  Windows của bạn bằng .venv_vintern đã chạy thành công.
