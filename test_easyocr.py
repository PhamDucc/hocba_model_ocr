from pathlib import Path
import torch
import easyocr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
image_path = ROOT / "data/crops/noi_sinh_that_001.png"

print("Đang đọc ảnh:", image_path)
image = np.array(Image.open(image_path).convert("RGB"))

# Ảnh đã cắt một dòng nên chỉ chạy nhận dạng.
reader = easyocr.Reader(["vi", "en"], gpu=False, detector=False)

results = reader.recognize(image, detail=1)

for _, text, confidence in results:
    print("Kết quả EasyOCR:", text)
    print("Điểm tin cậy:", confidence)