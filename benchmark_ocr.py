from pathlib import Path
import csv
import gc
import json
import unicodedata

import torch  # Phải đứng trước import Paddle trên Windows
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    # Bỏ dấu câu, giữ chữ, chữ số và dấu tiếng Việt.
    text = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    )
    return " ".join(text.split())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (a != b),
            ))
        previous = current
    return previous[-1]


def load_model(name):
    if name == "VietOCR":
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        config = Cfg.load_config_from_name("vgg_seq2seq")
        config["device"] = "cpu"
        config["cnn"]["pretrained"] = False
        config["predictor"]["beamsearch"] = False
        return Predictor(config)

    if name == "PaddleOCR":
        from paddleocr import TextRecognition

        return TextRecognition(
            model_name="latin_PP-OCRv5_mobile_rec",
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=2,
        )

    if name == "EasyOCR":
        import easyocr

        return easyocr.Reader(
            ["vi", "en"],
            gpu=False,
            detector=False,
        )

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    checkpoint = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint,
        use_safetensors=True,
    )
    model.eval()
    return processor, model


def recognize(name, model, path):
    with Image.open(path) as source:
        image = source.convert("RGB")

    if name == "VietOCR":
        return model.predict(image)

    if name == "PaddleOCR":
        results = list(model.predict(input=str(path)))
        return " ".join(str(r["rec_text"]) for r in results)

    if name == "EasyOCR":
        results = model.recognize(np.array(image), detail=1)
        return " ".join(r[1] for r in results)

    processor, network = model
    pixels = processor(images=image, return_tensors="pt").pixel_values
    with torch.inference_mode():
        ids = network.generate(pixels, max_new_tokens=128)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def main():
    labels_path = ROOT / "data/labels/ground_truth.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))

    samples = []
    for filename, text in labels.items():
        if not normalize(text):
            print("Bỏ qua nhãn trống:", filename)
            continue
        path = ROOT / "data/crops" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh: {path}")
        samples.append((filename, path, text))

    if not samples:
        raise ValueError("Chưa có ảnh với nhãn hợp lệ.")

    rows = []
    summary = []

    for name in ["VietOCR", "PaddleOCR", "EasyOCR", "TrOCR"]:
        print(f"\n===== {name} =====", flush=True)
        model = None
        errors = characters = succeeded = 0

        try:
            model = load_model(name)
            for filename, path, truth in samples:
                try:
                    prediction = recognize(name, model, path)
                    reference = normalize(truth)
                    distance = edit_distance(
                        reference, normalize(prediction)
                    )
                    cer = distance / len(reference)

                    errors += distance
                    characters += len(reference)
                    succeeded += 1

                    rows.append({
                        "model": name,
                        "image": filename,
                        "ground_truth": truth,
                        "prediction": prediction,
                        "cer_percent": round(cer * 100, 2),
                        "error": "",
                    })
                    print(
                        f"{filename}\n"
                        f"  Đọc: {prediction}\n"
                        f"  CER: {cer:.2%}",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append({
                        "model": name,
                        "image": filename,
                        "ground_truth": truth,
                        "prediction": "",
                        "cer_percent": "",
                        "error": str(exc),
                    })
                    print("Lỗi ảnh:", filename, exc, flush=True)

        except Exception as exc:
            print("Không nạp được model:", exc, flush=True)

        finally:
            del model
            gc.collect()

        # Chỉ công bố CER tổng khi model chạy đủ cùng bộ ảnh.
        complete = succeeded == len(samples)
        overall = errors / characters if complete else None
        summary.append({
            "model": name,
            "completed": succeeded,
            "total": len(samples),
            "cer_percent": (
                round(overall * 100, 2) if overall is not None else None
            ),
        })

        # Lưu sau mỗi model để giữ kết quả nếu lần sau bị gián đoạn.
        with (OUTPUT / "comparison.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=[
                "model", "image", "ground_truth",
                "prediction", "cer_percent", "error",
            ])
            writer.writeheader()
            writer.writerows(rows)

        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n===== TỔNG KẾT =====")
    for item in summary:
        score = item["cer_percent"]
        text = f"{score:.2f}%" if score is not None else "Chưa chạy đủ"
        print(
            f'{item["model"]}: CER {text} '
            f'({item["completed"]}/{item["total"]} ảnh)'
        )

    print("\nKết quả lưu tại:", OUTPUT)


if __name__ == "__main__":
    main()