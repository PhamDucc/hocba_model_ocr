from pathlib import Path
import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_ID = "5CD-AI/Vintern-1B-v3_5"


def load_image(path, max_tiles=6, size=448):
    transform = T.Compose([
        T.Resize(
            (size, size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    with Image.open(path) as source:
        image = source.convert("RGB")

    width, height = image.size
    aspect = width / height

    ratios = sorted(
        {
            (cols, rows)
            for cols in range(1, max_tiles + 1)
            for rows in range(1, max_tiles + 1)
            if cols * rows <= max_tiles
        },
        key=lambda pair: (pair[0] * pair[1], pair[0]),
    )

    best = (1, 1)
    best_difference = float("inf")

    for cols, rows in ratios:
        difference = abs(aspect - cols / rows)

        if difference < best_difference:
            best = (cols, rows)
            best_difference = difference
        elif difference == best_difference:
            if width * height > 0.5 * size * size * cols * rows:
                best = (cols, rows)

    cols, rows = best
    resized = image.resize((cols * size, rows * size))

    tiles = []
    for row in range(rows):
        for col in range(cols):
            tiles.append(resized.crop((
                col * size,
                row * size,
                (col + 1) * size,
                (row + 1) * size,
            )))

    if len(tiles) > 1:
        tiles.append(image.resize((size, size)))

    return torch.stack([transform(tile) for tile in tiles])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    torch.set_num_threads(min(8, os.cpu_count() or 1))

    print("Đang tải/nạp Vintern gốc...", flush=True)

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

    pixels = load_image(path).to(
        device="cpu",
        dtype=torch.float32,
    )

    print("Thiết bị:", next(model.parameters()).device)
    print("Đang đọc:", path)
    print("Số ô ảnh:", len(pixels), flush=True)

    question = (
        "<image>\n"
        "Đọc nguyên văn tất cả chữ trong ảnh. "
        "Chỉ trả về văn bản, không giải thích."
    )

    started = time.perf_counter()

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

    elapsed = time.perf_counter() - started

    print("\nKẾT QUẢ:")
    print(prediction)
    print(f"\nThời gian đọc: {elapsed:.1f} giây")

    destination = ROOT / "results" / "vintern_original"
    destination.mkdir(parents=True, exist_ok=True)

    output = destination / f"{path.stem}.txt"
    output.write_text(prediction, encoding="utf-8")
    print("Đã lưu:", output)


if __name__ == "__main__":
    main()