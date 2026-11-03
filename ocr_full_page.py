from pathlib import Path
import json

import torch  # Nạp trước Paddle để tránh lỗi DLL trên Windows
import cv2
import numpy as np
from PIL import Image, ImageDraw

from paddleocr import TextDetection
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/20261102_140620"
IMAGE_PATH = ROOT / "data/raw/synthetic_010.png"
OUTPUT = ROOT / "results/full_page" / IMAGE_PATH.stem


def crop_polygon(image, polygon):
    # Paddle trả bốn góc: trên trái, trên phải,
    # dưới phải, dưới trái.
    points = np.asarray(polygon, dtype=np.float32)

    width = max(
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[3]),
    )
    height = max(
        np.linalg.norm(points[3] - points[0]),
        np.linalg.norm(points[2] - points[1]),
    )

    width = max(2, int(round(width)))
    height = max(2, int(round(height)))

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(points, destination)

    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Thêm viền trắng nhỏ để chữ không sát mép.
    return cv2.copyMakeBorder(
        crop, 4, 4, 4, 4,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    crop_dir = OUTPUT / "crops"
    crop_dir.mkdir(exist_ok=True)

    torch.set_num_threads(2)

    print("Đang nạp detector PaddleOCR trên CPU...")
    detector = TextDetection(
        model_name="PP-OCRv5_mobile_det",
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=2,
    )

    print("Đang nạp VietOCR đã fine-tune trên CPU...")
    config = Cfg.load_config_from_file(str(RUN / "config.yml"))
    config["device"] = "cpu"
    config["weights"] = str(RUN / "best_val_loss.pth")
    config["cnn"]["pretrained"] = False
    config["predictor"]["beamsearch"] = False

    recognizer = Predictor(config)
    recognizer.model.eval()

    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")

    image_array = np.array(image)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    print("Đang tìm vùng chữ...")
    detection = list(detector.predict(input=str(IMAGE_PATH)))[0]
    polygons = detection["dt_polys"]

    # Thứ tự gần đúng: từ trên xuống dưới.
        # Nhóm các vùng chữ có tâm gần nhau theo chiều dọc.
    rows = []

    for polygon in sorted(
        polygons,
        key=lambda p: float(np.mean(np.asarray(p)[:, 1])),
    ):
        points = np.asarray(polygon)
        center_y = float(np.mean(points[:, 1]))
        height = float(np.ptp(points[:, 1]))

        if rows:
            row = rows[-1]
            tolerance = 0.5 * min(row["height"], height)

            if abs(center_y - row["center_y"]) <= tolerance:
                row["polygons"].append(polygon)
                row["center_y"] = float(np.mean([
                    np.mean(np.asarray(p)[:, 1])
                    for p in row["polygons"]
                ]))
                continue

        rows.append({
            "center_y": center_y,
            "height": height,
            "polygons": [polygon],
        })

    # Đọc từng hàng từ trái sang phải.
    polygons = [
        polygon
        for row in rows
        for polygon in sorted(
            row["polygons"],
            key=lambda p: float(np.min(np.asarray(p)[:, 0])),
        )
    ]

    records = []

    for index, polygon in enumerate(polygons, start=1):
        crop = crop_polygon(image_array, polygon)
        crop_name = f"line_{index:03d}.png"
        crop_image = Image.fromarray(crop)
        crop_image.save(crop_dir / crop_name)

        with torch.inference_mode():
            text = recognizer.predict(crop_image)

        points = [
            (int(x), int(y))
            for x, y in np.asarray(polygon)
        ]

        draw.line(
            points + [points[0]],
            fill="red",
            width=2,
        )
        draw.text(points[0], str(index), fill="blue")

        records.append({
            "index": index,
            "text": text,
            "polygon": np.asarray(polygon).tolist(),
            "crop": f"crops/{crop_name}",
        })

        print(f"{index:03d}: {text}", flush=True)

    overlay.save(OUTPUT / "boxes.jpg")

    (OUTPUT / "ocr.json").write_text(
        json.dumps({
            "source": IMAGE_PATH.name,
            "device": "cpu",
            "regions": records,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT / "ocr.txt").write_text(
        "\n".join(record["text"] for record in records),
        encoding="utf-8",
    )

    print(f"\nĐã xử lý {len(records)} vùng chữ.")
    print("Kết quả:", OUTPUT)


if __name__ == "__main__":
    main()