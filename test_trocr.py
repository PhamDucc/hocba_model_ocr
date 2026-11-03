from pathlib import Path
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parent
image_path = ROOT / "data/crops/noi_sinh_that_001.png"
model_name = "microsoft/trocr-base-handwritten"

print("Đang đọc ảnh:", image_path)
image = Image.open(image_path).convert("RGB")

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(
    model_name,
    use_safetensors=True,
)
model.eval()

pixel_values = processor(
    images=image,
    return_tensors="pt",
).pixel_values

with torch.inference_mode():
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=128,
    )

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print("Kết quả TrOCR:", text)