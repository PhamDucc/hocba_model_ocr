from pathlib import Path
import argparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Đường dẫn ảnh học bạ")
    args = parser.parse_args()

    source = Path(args.image)
    if not source.is_absolute():
        source = ROOT / source

    if not source.is_file():
        raise FileNotFoundError(source)

    # Hỗ trợ đường dẫn Windows có dấu tiếng Việt.
    image = cv2.imdecode(
        np.fromfile(str(source), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {source}")

    height, width = image.shape[:2]

    # Giữ toàn bộ chiều ngang; cắt theo tỷ lệ chiều cao.
    regions = {
        "01_dau_trang": (0.00, 0.30),
        "02_thong_tin": (0.20, 0.70),
        "03_qua_trinh_hoc": (0.60, 1.00),
    }

    output = ROOT / "results" / "vintern" / source.stem
    output.mkdir(parents=True, exist_ok=True)

    for name, (start, end) in regions.items():
        crop = image[int(start * height):int(end * height), :width]
        destination = output / f"{name}.png"

        success, encoded = cv2.imencode(".png", crop)
        if not success:
            raise RuntimeError(f"Không lưu được: {destination}")

        encoded.tofile(str(destination))
        print("Đã lưu:", destination)


if __name__ == "__main__":
    main()
    