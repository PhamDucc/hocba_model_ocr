from pathlib import Path
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

ROOT = Path(__file__).resolve().parent
image_path = ROOT / "data/crops/noi_sinh_that_001.png"
print("Đang đọc ảnh:", image_path)

if not image_path.exists():
    raise FileNotFoundError(f"Chưa tìm thấy ảnh: {image_path}")

config = Cfg.load_config_from_name("vgg_seq2seq")
config["device"] = "cpu"
config["cnn"]["pretrained"] = False
config["predictor"]["beamsearch"] = False

model = Predictor(config)
image = Image.open(image_path).convert("RGB")

text = model.predict(image)
print("Kết quả VietOCR:", text)