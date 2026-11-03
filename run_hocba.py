from pathlib import Path
import argparse
import importlib.util
import json

ROOT = Path(__file__).resolve().parent


def load_script(name):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Không nạp được: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="OCR trang đầu học bạ, chạy hoàn toàn bằng CPU."
    )
    parser.add_argument(
        "image",
        help="Đường dẫn ảnh học bạ đầu vào.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    image_path = image_path.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    output = ROOT / "results/full_page" / image_path.stem

    print("\n[1/3] Phát hiện vùng chữ và nhận dạng...", flush=True)
    ocr = load_script("ocr_full_page")
    ocr.IMAGE_PATH = image_path
    ocr.OUTPUT = output
    ocr.main()

    print("\n[2/3] Trích xuất thông tin cá nhân...", flush=True)
    fields = load_script("extract_fields")
    fields.FOLDER = output
    fields.main()

    print("\n[3/3] Trích xuất quá trình học tập...", flush=True)
    history = load_script("extract_history")
    history.FOLDER = output

    history_error = None

    try:
        history.main()
    except ValueError as exc:
        # Giữ kết quả thông tin cá nhân nếu chưa đọc được bảng.
        history_error = str(exc)
        student_path = output / "student.json"
        student = json.loads(
            student_path.read_text(encoding="utf-8")
        )
        student["qua_trinh_hoc_tap"] = None
        student["history_error"] = history_error
        student_path.write_text(
            json.dumps(student, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n===== HOÀN THÀNH =====")
    print("Ảnh đầu vào:", image_path)
    print("JSON kết quả:", output / "student.json")
    print("Ảnh khoanh vùng:", output / "boxes.jpg")

    if history_error:
        print("Bảng học tập cần kiểm tra:", history_error)


if __name__ == "__main__":
    main()