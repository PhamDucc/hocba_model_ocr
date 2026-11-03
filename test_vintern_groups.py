from pathlib import Path
import os
import time
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from test_vintern_cpu import load_image, MODEL_ID


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/user_003.png"
OUTPUT = ROOT / "results/vintern_groups/user_003"


def main():
    global SOURCE, OUTPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--crop-only", action="store_true")
    args = parser.parse_args()

    SOURCE = Path(args.image)
    if not SOURCE.is_absolute():
        SOURCE = ROOT / SOURCE
    SOURCE = SOURCE.resolve()

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    OUTPUT = ROOT / "results/vintern_groups" / SOURCE.stem
    OUTPUT.mkdir(parents=True, exist_ok=True)

    (OUTPUT / "source.json").write_text(
        json.dumps(
            {"source": SOURCE.name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if SOURCE.stem == "user_002":
        groups = {
            "01_ca_nhan": (0.02, 0.235, 0.98, 0.38),
            "02_dia_chi": (0.02, 0.405, 0.98, 0.445),
            "03_cha_me": (0.02, 0.435, 0.98, 0.515),
            "04_gioi_tinh": (0.47, 0.235, 0.75, 0.28),
        }
    elif SOURCE.stem == "user_003":
        groups = {
            "01_ca_nhan": (0.07, 0.23, 0.90, 0.33),
            "02_dia_chi": (0.07, 0.36, 0.91, 0.41),
            "03_cha_me": (0.07, 0.405, 0.92, 0.46),
            "04_gioi_tinh": (0.66, 0.22, 0.85, 0.26),
        }
    else:
        raise ValueError(
            "Ảnh này chưa có cấu hình vùng cắt."
        )

    with Image.open(SOURCE) as source:
        image = source.convert("RGB")
        width, height = image.size

        for name, (left, top, right, bottom) in groups.items():
            image.crop((
                int(left * width),
                int(top * height),
                int(right * width),
                int(bottom * height),
            )).save(OUTPUT / f"{name}.png")

    print(f"Đã lưu {len(groups)} ảnh nhóm tại:", OUTPUT)
    if args.crop_only:
        print("Chỉ cắt ảnh, chưa chạy OCR.")
        return
    print("Đang nạp Vintern trên CPU...", flush=True)

    torch.set_num_threads(min(8, os.cpu_count() or 1))

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).to("cpu").eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    question = (
        "<image>\n"
        "Chép nguyên văn tất cả các dòng chữ trong ảnh, "
        "theo thứ tự từ trên xuống dưới. "
        "Không tóm tắt, không tự bổ sung thông tin."
    )

    total_seconds = 0
    reports = []

    for name in groups:
        print(f"\n===== {name} =====", flush=True)
        started = time.perf_counter()

        pixels = load_image(OUTPUT / f"{name}.png").to(
            device="cpu",
            dtype=torch.float32,
        )

        with torch.inference_mode():
            prediction = model.chat(
                tokenizer,
                pixels,
                question,
                {
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "num_beams": 1,
                },
            )

        seconds = time.perf_counter() - started
        total_seconds += seconds

        print(prediction, flush=True)
        print(f"Thời gian: {seconds:.1f} giây", flush=True)

        (OUTPUT / f"{name}.txt").write_text(
            prediction, encoding="utf-8"
        )
        reports.append(
            f"===== {name} =====\n"
            f"{prediction}\n"
            f"Thời gian: {seconds:.1f} giây\n"
        )

        del pixels

    reports.append(
        f"Tổng thời gian đọc 3 nhóm: {total_seconds:.1f} giây"
    )
    (OUTPUT / "summary.txt").write_text(
        "\n".join(reports), encoding="utf-8"
    )

    print(
    f"\nTổng thời gian đọc {len(groups)} nhóm: "
    f"{total_seconds:.1f} giây"
)
    print("Chưa bao gồm thời gian nạp model.")


if __name__ == "__main__":
    main()