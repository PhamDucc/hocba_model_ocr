from logging import config
from pathlib import Path
from datetime import datetime
import unicodedata
import os

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from vietocr.tool.config import Cfg
from vietocr.tool.translate import process_image
from vietocr.loader.dataloader import Collator
from vietocr.model.trainer import Trainer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_labels(filename, expected_folder):
    path = DATA / filename
    samples = []

    for number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"{filename}, dòng {number}: cần đúng 1 Tab "
                "giữa đường dẫn và nội dung."
            )

        relative, text = parts
        relative = relative.strip().replace("\\", "/")
        text = unicodedata.normalize("NFC", text.strip())

        if not relative.startswith(expected_folder + "/"):
            raise ValueError(
                f"{filename}, dòng {number}: đường dẫn phải "
                f"bắt đầu bằng {expected_folder}/"
            )

        image_path = DATA / relative
        if not text:
            raise ValueError(f"Nhãn trống: {relative}")

        with Image.open(image_path) as image:
            image.verify()

        samples.append((str(image_path), text))

    if not samples:
        raise ValueError(f"{filename} không có dữ liệu.")

    return samples


class LineDataset(Dataset):
    def __init__(self, samples, vocab, config):
        self.samples = samples
        self.vocab = vocab
        self.config = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, text = self.samples[index]

        with Image.open(path) as source:
            image = source.convert("RGB")

        image = process_image(
            image,
            self.config["image_height"],
            self.config["image_min_width"],
            self.config["image_max_width"],
        )

        return {
            "img": image,
            "word": self.vocab.encode(text),
            "img_path": path,
        }


class CPUTrainer(Trainer):
    def step(self, batch):
        # Trainer.step() sẽ gọi model.train().
        # Hook đặt BatchNorm về eval ngay trước mỗi forward.
        def keep_batchnorm_eval(module, inputs):
            for layer in module.modules():
                if isinstance(
                    layer,
                    torch.nn.modules.batchnorm._BatchNorm,
                ):
                    layer.eval()

        handle = self.model.register_forward_pre_hook(
            keep_batchnorm_eval
        )
        try:
            return super().step(batch)
        finally:
            handle.remove()    
    # Dùng ảnh trực tiếp, tránh tạo bộ nhớ đệm LMDB.
    def data_gen(
        self, lmdb_path, data_root, annotation,
        masked_language_model=True, transform=None
    ):
        samples = (
            self.train_samples
            if annotation == "train_labels.txt"
            else self.val_samples
        )

        dataset = LineDataset(
            samples, self.vocab, self.config["dataset"]
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=(annotation == "train_labels.txt"),
            num_workers=0,
            pin_memory=False,
            collate_fn=Collator(masked_language_model=False),
        )

    def validate(self):
        loss = super().validate()

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.save_weights(str(self.output / "best_val_loss.pth"))
            print(f"Đã lưu best_val_loss.pth — loss: {loss:.4f}")

        return loss

    def precision(self, sample=None):
        self.model.eval()
        try:
            return super().precision(sample)
        finally:
            self.model.train()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)

    train_samples = read_labels("train_labels.txt", "train")
    val_samples = read_labels("val_labels.txt", "val")

    train_pages = {
        Path(path).stem.split("_")[0] for path, _ in train_samples
    }
    val_pages = {
        Path(path).stem.split("_")[0] for path, _ in val_samples
    }

    if train_pages & val_pages:
        raise ValueError("Có cùng trang xuất hiện trong train và val.")

    output = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "runs/20261102_140620/config.yml"

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = Cfg.load_config_from_file(str(config_path))
    config["device"] = "cpu"
    config["cnn"]["pretrained"] = False
    config["predictor"]["beamsearch"] = False

    for path, text in train_samples + val_samples:
        unknown = set(text) - set(config["vocab"])
        if unknown:
            raise ValueError(f"{path}: ký tự ngoài vocab: {unknown}")

    config["dataset"].update({
        "name": "hocba_cpu",
        "data_root": str(DATA),
        "train_annotation": "train_labels.txt",
        "valid_annotation": "val_labels.txt",
    })

    config["trainer"].update({
        "batch_size": 1,
        "iters": 100,
        "print_every": 10,
        "valid_every": 20,
        "metrics": None,
        "export": str(output / "best_accuracy.pth"),
        "checkpoint": str(output / "checkpoint.pth"),
        "log": str(output / "train.log"),
    })

    config["optimizer"].update({
        "max_lr": 1e-5,
        "pct_start": 0.1,
    })

    config["dataloader"].update({
        "num_workers": 0,
        "pin_memory": False,
    })

    config["aug"]["image_aug"] = False
    config["aug"]["masked_language_model"] = False

    CPUTrainer.train_samples = train_samples
    CPUTrainer.val_samples = val_samples

    # Các tệp phụ của thư viện nằm riêng trong thư mục lần chạy.
    os.chdir(output)

    trainer = CPUTrainer(config, pretrained=True)
    trainer.output = output
    trainer.best_val_loss = float("inf")

    trainer.save_weights(str(output / "initial.pth"))

    print(f"\nThiết bị: {next(trainer.model.parameters()).device}")
    print(f"Train: {len(train_samples)} ảnh")
    print(f"Validation: {len(val_samples)} ảnh")
    print(f"Lưu kết quả tại: {output}\n")

    print("Đánh giá validation trước fine-tune:")
    trainer.validate()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu.")
    finally:
        trainer.save_weights(str(output / "last.pth"))
        config["weights"] = str(output / "best_val_loss.pth")
        config.save(str(output / "config.yml"))
        print("\nĐã lưu trọng số và cấu hình tại:", output)


if __name__ == "__main__":
    main()