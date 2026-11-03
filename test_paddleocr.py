from pathlib import Path
import torch  # Nạp trước Paddle để tránh lỗi DLL trên Windows
from paddleocr import TextRecognition

ROOT = Path(__file__).resolve().parent
image_path = ROOT / "data/crops/noi_sinh_that_001.png"
print("Đang đọc ảnh:", image_path)

if not image_path.exists():
    raise FileNotFoundError(f"Chưa tìm thấy ảnh: {image_path}")

model = TextRecognition(
    model_name="latin_PP-OCRv5_mobile_rec",
    device="cpu",
    enable_mkldnn=False,
    cpu_threads=2,
)

for result in model.predict(input=str(image_path)):
    print("Kết quả PaddleOCR:", result["rec_text"])
    print("Điểm tin cậy:", result["rec_score"])