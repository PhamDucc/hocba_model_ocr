from pathlib import Path
import argparse
import json
import os
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModel, AutoTokenizer

from test_vintern_cpu import load_image, MODEL_ID


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        help="Thư mục kết quả Paddle có ocr.json và crops/",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = ROOT / folder

    metadata = folder / "ocr.json"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)

    original = json.loads(
        metadata.read_text(encoding="utf-8-sig")
    )
    regions = original.get("regions", [])

    if not regions:
        raise ValueError("ocr.json không có vùng chữ.")

    # Kiểm tra đủ ảnh trước khi nạp model.
    for region in regions:
        path = folder / region["crop"]
        if not path.is_file():
            raise FileNotFoundError(path)

    output = ROOT / "results" / "vintern_full_page" / folder.name
    (output / "crops").mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(min(8, os.cpu_count() or 1))

    print("Đang nạp Vintern gốc trên CPU...", flush=True)
    load_started = time.perf_counter()

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

    load_seconds = time.perf_counter() - load_started
    print(f"Nạp model: {load_seconds:.1f} giây")
    print(f"Số vùng cần đọc: {len(regions)}", flush=True)

    question = (
        "<image>\n"
        "Đọc nguyên văn tất cả chữ trong ảnh. "
        "Chỉ trả về văn bản, không giải thích."
    )
    generation = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
    }

    results = []
    started = time.perf_counter()

    for position, region in enumerate(regions, 1):
        path = folder / region["crop"]
        print(
            f"\n[{position}/{len(regions)}] {path.name}",
            flush=True,
        )

        sample_started = time.perf_counter()
        pixels = load_image(path).to(
            device="cpu",
            dtype=torch.float32,
        )

        with torch.inference_mode():
            prediction = model.chat(
                tokenizer,
                pixels,
                question,
                dict(generation),
            )

        seconds = time.perf_counter() - sample_started
        print(prediction, flush=True)
        print(f"Thời gian: {seconds:.1f} giây", flush=True)

        # Giữ index, polygon và ảnh để tra lại nguồn.
        crop_name = f"line_{int(region['index']):03d}.png"
        shutil.copy2(path, output / "crops" / crop_name)

        item = dict(region)
        item["text"] = prediction.strip()
        item["crop"] = f"crops/{crop_name}"
        item["previous_text"] = region.get("text", "")
        item["seconds"] = round(seconds, 2)
        results.append(item)

        # Lưu tiến độ sau mỗi vùng.
        partial = {
            "source": original.get("source", folder.name),
            "device": "cpu",
            "model": MODEL_ID,
            "completed": len(results),
            "total": len(regions),
            "regions": results,
        }
        (output / "ocr.partial.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        del pixels

    reading_seconds = time.perf_counter() - started

    final = {
        "source": original.get("source", folder.name),
        "device": "cpu",
        "model": MODEL_ID,
        "load_seconds": round(load_seconds, 2),
        "reading_seconds": round(reading_seconds, 2),
        "regions": results,
    }

    (output / "ocr.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "ocr.txt").write_text(
        "\n".join(item["text"] for item in results),
        encoding="utf-8",
    )
    (output / "ocr.partial.json").unlink(missing_ok=True)

    print(f"\nĐã đọc {len(results)} vùng.")
    print(f"Tổng thời gian đọc: {reading_seconds:.1f} giây")
    print("Kết quả tại:", output)


if __name__ == "__main__":
    main()