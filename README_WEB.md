# Web demo học bạ OCR trên CPU

## Cài vào dự án hiện có
Sao chép app.py, requirements_web.txt và thư mục templates vào D:\VNPT - AI\hocba_ocr. Không đặt chúng trong thư mục con hocba_web.
Không thay bốn script OCR đã có. Không thay môi trường Python hay trọng số.

Cấu trúc:
```
hocba_ocr/
  app.py
  requirements_web.txt
  templates/index.html
  run_hocba.py
  ocr_full_page.py
  extract_fields.py
  extract_history.py
  runs/20261102_140620/config.yml
  runs/20261102_140620/best_val_loss.pth
```

## Chạy localhost — PowerShell tại thư mục hocba_ocr
```
.\.venv310\Scripts\python.exe -m pip install -r requirements_web.txt
.\.venv310\Scripts\python.exe app.py
```
Mở http://127.0.0.1:5000. Chọn ảnh synthetic_010.png và bấm Đọc học bạ.
Ứng dụng dùng môi trường Python đang chạy web để gọi run_hocba.py. CPU và checkpoint giữ nguyên theo ocr_full_page.py hiện tại.

## Mở đường dẫn ngrok
1. Cài ngrok từ https://ngrok.com/download/windows hoặc dùng Microsoft Store:
```
winget install ngrok -s msstore
```
Mở terminal mới, kiểm tra `ngrok version`.
2. Đăng nhập tài khoản ngrok. Lấy authtoken của chính mình tại https://dashboard.ngrok.com/get-started/your-authtoken . Chạy lệnh `ngrok config add-authtoken` hiển thị trên trang đó, với token thật. Không nhập nguyên chữ giữ chỗ và không gửi token cho người khác.
3. Giữ terminal web mở, dùng terminal thứ hai:
```
ngrok http 5000
```
4. Copy địa chỉ HTTPS ở dòng Forwarding gửi cho mentor. Giữ máy, kết nối Internet và cả hai terminal hoạt động. Link công khai cho phép người truy cập tải ảnh lên; dùng dữ liệu mô phỏng cho buổi demo. Ctrl+C ngrok khi kết thúc.

## Cách hoạt động
- Một ảnh được xử lý tại một thời điểm để tránh tranh CPU; lượt khác nhận thông báo bận.
- Xử lý chạy nền và giao diện hỏi trạng thái định kỳ, tránh giữ một HTTP request OCR quá lâu qua ngrok.
- Mỗi lượt có mã ngẫu nhiên, ảnh được chuẩn hóa thành PNG. Không dùng tên file tải lên làm lệnh hoặc đường dẫn hệ thống.
- Giới hạn 12 MB/20 megapixel; nhận JPG, PNG, WEBP. Chưa hỗ trợ PDF.
- Kết quả nằm tại results/full_page/<mã-lượt>/; ảnh tải lên và process.log tại web_uploads/<mã-lượt>/.
- Script OCR hiện tại nạp model lại mỗi lượt, nên có thể chậm. Chưa tối ưu giữ model trong RAM.
- Không tự sửa chữ OCR và không xác nhận dữ liệu là đúng. Cảnh báo ngày sinh và lỗi bảng được giữ nguyên.
- Không ghi đè kết quả demo synthetic_001…synthetic_010 vì ảnh tải lên dùng tên mã riêng.
- Lượt xử lý/đường dẫn kết quả web chỉ có hiệu lực trong phiên máy chủ; khởi động lại web sẽ xóa danh sách lượt trong RAM. Các tệp trên đĩa vẫn còn và chưa có cơ chế tự xóa.

## Lỗi thường gặp
- Thiếu tệp: đặt app.py đúng thư mục gốc dự án, giữ đủ checkpoint và config.
- OCR lỗi: mở web_uploads/<mã-lượt>/process.log hoặc terminal web. Dùng lại lệnh run_hocba.py để kiểm tra luồng độc lập.
- Không chạy được ngrok: cài xong mở PowerShell mới, hoặc dùng đường dẫn đầy đủ tới ngrok.exe.
- Cổng 5000 đã dùng: dừng ứng dụng cũ trên cổng đó trước khi chạy lại.

## Kiểm tra thực hiện khi đóng gói
Đã kiểm tra cú pháp Python, giao diện và các nhánh Flask bằng bộ OCR giả lập: tải ảnh, xử lý nền, lấy kết quả, tải JSON, từ chối tệp sai, báo bận và xử lý lỗi. Chưa chạy model thật/Windows/ngrok trong gói web vì trọng số và môi trường dự án ở máy người dùng.
Nguồn hướng dẫn ngrok: https://ngrok.com/download/windows và https://ngrok.com/docs/gateway/endpoints/agent-cli-quickstart .
