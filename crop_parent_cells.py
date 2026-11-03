from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/user_002.png"
OUTPUT = ROOT / "results/vintern_cells/user_002"

# Tọa độ thử riêng cho ảnh user_002.
REGIONS = {
    "ho_ten_cha": (0.02, 0.44, 0.57, 0.48),
    "ho_ten_me": (0.02, 0.475, 0.57, 0.515),
    "nghe_nghiep_cha": (0.57, 0.425, 0.98, 0.465),
    "nghe_nghiep_me": (0.57, 0.46, 0.98, 0.50),
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with Image.open(SOURCE) as source:
        image = source.convert("RGB")
        width, height = image.size

        for name, (left, top, right, bottom) in REGIONS.items():
            crop = image.crop((
                int(left * width),
                int(top * height),
                int(right * width),
                int(bottom * height),
            ))

            destination = OUTPUT / f"{name}.png"
            crop.save(destination)
            print("Đã lưu:", destination)


if __name__ == "__main__":
    main()