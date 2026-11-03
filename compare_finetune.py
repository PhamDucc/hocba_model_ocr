from pathlib import Path
import csv
import json
import unicodedata

from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/20261103_085912"
DATA = ROOT / "data"


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
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


def main():
    labels = json.loads(
        (DATA / "labels/ground_truth.json").read_text(
            encoding="utf-8-sig"
        )
    )

    samples = []

    for filename, truth in labels.items():
        path = DATA / "crops" / "baseline_5_crops" / filename
        truth = truth.strip()

        if not path.is_file():  
            raise FileNotFoundError(path)

        if not normalize(truth):
            raise ValueError(f"Nhãn trống: {path}")

        samples.append((path, truth))

    if not samples:
        raise ValueError("Không có ảnh thật để đánh giá.")

    results = []
    summaries = []

    checkpoints = [
        "initial.pth",
        "best_val_loss.pth",
        "last.pth",
    ]

    for checkpoint in checkpoints:
        print(f"\n===== {checkpoint} =====", flush=True)

        config = Cfg.load_config_from_file(
            str(RUN / "config.yml")
        )
        config["device"] = "cpu"
        config["weights"] = str(RUN / checkpoint)
        config["cnn"]["pretrained"] = False
        config["predictor"]["beamsearch"] = False

        predictor = Predictor(config)
        predictor.model.eval()

        total_errors = 0
        total_chars = 0
        exact_matches = 0

        for path, truth in samples:
            with Image.open(path) as image:
                prediction = predictor.predict(
                    image.convert("RGB")
                )

            reference = normalize(truth)
            hypothesis = normalize(prediction)

            errors = edit_distance(reference, hypothesis)
            cer = errors / len(reference)

            total_errors += errors
            total_chars += len(reference)
            exact_matches += reference == hypothesis

            print(f"\nẢnh: {path.name}")
            print("Đúng:", truth)
            print("Đọc: ", prediction)
            print(f"CER: {cer:.2%}", flush=True)

            results.append({
                "checkpoint": checkpoint,
                "image": path.name,
                "ground_truth": truth,
                "prediction": prediction,
                "cer_percent": round(cer * 100, 2),
            })

        summaries.append(
            f"{checkpoint}: CER "
            f"{total_errors / total_chars:.2%}"
            f" — đúng cả dòng "
            f"{exact_matches}/{len(samples)}"
        )

        del predictor

    destination = RUN / "compare_real.csv"

    with destination.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint",
                "image",
                "ground_truth",
                "prediction",
                "cer_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== SO SÁNH TRÊN ẢNH THẬT =====")

    for summary in summaries:
        print(summary)

    print("\nĐã lưu:", destination)


if __name__ == "__main__":
    main()