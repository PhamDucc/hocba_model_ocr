# hocba_ocr — Bộ dữ liệu khởi đầu

## Cấu trúc
- data/raw/: ảnh nguyên trang, không bao gồm trang điểm.
- data/crops/: để trống, dành cho các vùng chữ sau khi cắt và kiểm tra.
- data/labels/: bảng kiểm kê và nhãn cần gán.
- prompts/: câu lệnh tạo ảnh mô phỏng để truy xuất nguồn gốc.

## Nguồn dữ liệu
- user_001.jpg: ảnh người dùng cung cấp, độ phân giải thấp (450×600).
- public_001.jpg: trích ảnh trong trang 28 của tài liệu hướng dẫn tuyển sinh công khai của HUNRE (2023). URL: https://hunre.edu.vn/download/news/2023/07/05/163809_2485-hd-hunre.pdf . Ảnh tham khảo, chưa xác minh quyền dùng cho huấn luyện/phân phối ngoài phạm vi thử nghiệm cá nhân; không coi là giấy phép dữ liệu mở.
- synthetic_*.png: ảnh tạo bởi AI, thông tin hư cấu, dấu mô phỏng; không phải bản ghi học sinh thật. Có dòng đánh dấu ở chân trang.

## Cách dùng
1. Kiểm tra thủ công từng ảnh, loại ảnh có chữ lỗi hoặc bố cục không hợp lệ.
2. Cắt riêng từng dòng/vùng: họ tên, ngày sinh, giới tính, nơi sinh, dân tộc.
3. Chép lại nội dung THỰC SỰ NHÌN THẤY trong ảnh. Câu lệnh tạo ảnh KHÔNG phải nhãn chuẩn vì AI có thể viết sai.
4. Lưu nhãn recognition theo dạng đường_dẫn_ảnh[TAB]văn_bản.
5. Chạy model gốc trước và lưu dự đoán, rồi mới fine-tune.

Đây là bộ thử quy trình, không phải bộ train/test đã gán nhãn hay đại diện cho chữ viết tay thực tế. Không dùng ảnh mô phỏng để công bố độ chính xác trên học bạ thật. Không tạo train/test ngẫu nhiên từ các biến thể cùng ảnh. Các ảnh thật cần chia theo quyển học bạ/học sinh.
