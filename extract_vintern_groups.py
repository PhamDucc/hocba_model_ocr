from pathlib import Path
import json
import argparse
import csv
from datetime import datetime
import re
import unicodedata


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "results/vintern_groups/user_003"


def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("đ", "d")


def read_lines(filename):
    path = FOLDER / filename
    return [
        line.strip().replace("**", "")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def get_value(lines, label):
    expected = normalize(label)

    for line in lines:
        parts = re.split(r"[:：]", line, maxsplit=1)
        if len(parts) == 2:
            key, value = parts
            if normalize(key.strip()) == expected:
                return value.strip() or None

    return None


def get_parent_jobs(lines):
    jobs = {"nghe_nghiep_cha": None, "nghe_nghiep_me": None}
    current_parent = None

    for line in lines:
        parts = re.split(r"[:：]", line, maxsplit=1)
        if len(parts) != 2:
            continue

        key, value = parts
        key = normalize(key.strip())
        value = value.strip() or None

        if key == "ho va ten cha":
            current_parent = "cha"
        elif key == "ho va ten me":
            current_parent = "me"
        elif key == "nghe nghiep" and current_parent:
            jobs[f"nghe_nghiep_{current_parent}"] = value

    return jobs


def main(folder=None):
    global FOLDER
    if folder is not None:
        FOLDER = Path(folder)
        if not FOLDER.is_absolute():
            FOLDER = ROOT / FOLDER

    personal = read_lines("01_ca_nhan.txt")
    address = read_lines("02_dia_chi.txt")
    parents = read_lines("03_cha_me.txt")

    fields = {
        "ho_ten": get_value(personal, "Họ và tên"),
        "gioi_tinh": get_value(personal, "Giới tính"),
        "ngay_sinh": get_value(personal, "Ngày sinh"),
        "noi_sinh": get_value(personal, "Nơi sinh"),
        "dan_toc": get_value(personal, "Dân tộc"),
        "dia_chi": get_value(address, "Chỗ ở hiện tại"),
        "ho_ten_cha": get_value(parents, "Họ và tên cha"),
        "ho_ten_me": get_value(parents, "Họ và tên mẹ"),
        **get_parent_jobs(parents),
    }

    # Ưu tiên kết quả đọc riêng vùng giới tính, nếu có.
    gender_path = FOLDER / "04_gioi_tinh.txt"

    if gender_path.is_file():
        gender_lines = gender_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        gender = get_value(gender_lines, "Giới tính")

        if gender:
            fields["gioi_tinh"] = gender

    source_info = json.loads(
        (FOLDER / "source.json").read_text(encoding="utf-8-sig")
    )
    source = source_info["source"]
    raw_fields = dict(fields)
    raw_groups = {
        name: (FOLDER / name).read_text(encoding="utf-8-sig")
        for name in (
            "01_ca_nhan.txt", "02_dia_chi.txt",
            "03_cha_me.txt", "04_gioi_tinh.txt",
        )
        if (FOLDER / name).is_file()
    }
    verified_path = FOLDER / "verified_fields.json"
    manual_verification = {}
    confirmed_fields = {}
    confirmed_by = None

    if verified_path.is_file():
        verified = json.loads(verified_path.read_text(encoding="utf-8-sig"))
        if verified.get("source") != source:
            raise ValueError("File xác nhận không khớp ảnh nguồn.")
        confirmed_fields = verified.get("fields")
        if not isinstance(confirmed_fields, dict):
            raise ValueError("fields trong file xác nhận phải là object JSON.")
        confirmed_by = verified.get("confirmed_by")
        if not isinstance(confirmed_by, str) or not confirmed_by.strip():
            raise ValueError("File xác nhận thiếu confirmed_by.")
        for name, value in confirmed_fields.items():
            if name not in fields:
                raise ValueError(f"Trường không hợp lệ: {name}")
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"Nhãn xác nhận phải là chuỗi có chữ hoặc null: {name}")
            manual_verification[name] = {
                "ocr_value": raw_fields[name],
                "value": value,
                "confirmed_by": confirmed_by,
            }
            fields[name] = value

    review = [
        {
            "field": name,
            "reason": (
                "Chưa lấy được từ kết quả OCR."
                if value is None
                else "Đã có kết quả OCR, chưa được xác minh."
            ),
        }
        for name, value in fields.items()
        if name not in manual_verification
    ]
    evaluated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    raw_result = {
        "source": source,
        "fields": raw_fields,
        "raw_groups": raw_groups,
        "verification_status": "unverified",
        "note": (
            "fields là giá trị trích xuất trước sửa thủ công; "
            "raw_groups giữ nguyên văn bản model sinh ra. "
            "null có thể do OCR hoặc bộ trích xuất không nhận ra nhãn."
        ),
    }
    result = {
        "source": source,
        "fields": fields,
        "manual_verification": manual_verification,
        "needs_review": review,
        "verification_status": (
            "verified" if not review
            else "partially_verified" if manual_verification
            else "unverified"
        ),
        "note": (
            "Kết quả đã áp dụng các xác nhận riêng của hồ sơ. "
            "Trường chưa xác nhận vẫn có thể sai hoặc thiếu chữ. "
            "Sửa thủ công không thay đổi trọng số model."
        ),
    }
    evaluation, rows = evaluate_fields(source, raw_fields, confirmed_fields)
    evaluation["evaluated_at"] = evaluated_at

    # Chỉ ghi sau khi đã kiểm tra nhãn xác nhận và tính thống kê thành công.
    destination = FOLDER / "student.json"
    if destination.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination.with_name(f"student.{stamp}.bak.json").write_bytes(
            destination.read_bytes()
        )
    write_json(FOLDER / "student_raw.json", raw_result)
    write_json(destination, result)
    write_json(FOLDER / "evaluation.json", evaluation)
    with (FOLDER / "evaluation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=[
            "field", "reference", "prediction", "status",
            "errors", "reference_chars", "cer_percent", "exact_match",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n===== ĐÁNH GIÁ TRÊN TRƯỜNG ĐÃ XÁC NHẬN =====")
    print(f"Đã xác nhận: {len(confirmed_fields)}/{len(raw_fields)} trường")
    print(f"Có đáp án văn bản để tính CER: {evaluation['text_reference_fields']}")
    cer = evaluation["cer_percent"]
    print("CER:", "Chưa có đáp án văn bản" if cer is None else f"{cer:.2f}%")
    print(
        "Đúng giá trị sau chuẩn hóa:",
        f"{evaluation['exact_matches']}/{evaluation['confirmed_fields']}",
    )
    print("Lưu ý: lỗi tính cả OCR và trích xuất trường, trước sửa thủ công.")
    for name in ("student_raw.json", "student.json", "evaluation.json", "evaluation.csv"):
        print("Đã lưu:", FOLDER / name)


def metric_normalize(text):
    # Giữ dấu tiếng Việt; không dùng normalize() dành cho so khớp tên nhãn.
    text = unicodedata.normalize("NFC", text or "").lower()
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


def evaluate_fields(source, raw_fields, confirmed_fields):
    rows = []
    total_errors = total_chars = exact_matches = text_fields = 0
    missing_text_fields = 0
    for name, prediction in raw_fields.items():
        row = {
            "field": name, "reference": None, "prediction": prediction,
            "status": "unverified", "errors": None,
            "reference_chars": None, "cer_percent": None, "exact_match": None,
        }
        if name in confirmed_fields:
            truth = confirmed_fields[name]
            row["reference"] = truth
            if truth is None:
                # null trong file xác nhận nghĩa là người kiểm tra xác nhận trống.
                exact = prediction is None or not str(prediction).strip()
                row.update(status="confirmed_empty", exact_match=exact)
                exact_matches += int(exact)
            else:
                reference = metric_normalize(truth)
                if not reference:
                    raise ValueError(f"Đáp án không còn ký tự sau chuẩn hóa: {name}")
                hypothesis = metric_normalize(prediction)
                errors = edit_distance(reference, hypothesis)
                exact = reference == hypothesis
                row.update(
                    status="confirmed_text", errors=errors,
                    reference_chars=len(reference),
                    cer_percent=round(100 * errors / len(reference), 2),
                    exact_match=exact,
                )
                total_errors += errors
                total_chars += len(reference)
                exact_matches += int(exact)
                text_fields += 1
                missing_text_fields += int(not hypothesis)
        rows.append(row)
    return {
        "source": source,
        "scope": "OCR + trích xuất trường, trước sửa thủ công; chỉ trường đã xác nhận",
        "normalization": "NFC, chữ thường, thay dấu câu bằng khoảng trắng, gộp khoảng trắng; giữ dấu tiếng Việt",
        "confirmed_fields": len(confirmed_fields),
        "unverified_fields": len(raw_fields) - len(confirmed_fields),
        "text_reference_fields": text_fields,
        "missing_text_fields": missing_text_fields,
        "exact_matches": exact_matches,
        "errors": total_errors,
        "reference_chars": total_chars,
        "cer_percent": round(100 * total_errors / total_chars, 2) if total_chars else None,
        "note": "Trường chưa xác nhận không tham gia chấm. Dự đoán null với đáp án có chữ tính là xóa toàn bộ ký tự. CER có thể lớn hơn 100%.",
    }, rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất, áp dụng xác nhận và đánh giá; không chạy OCR.")
    parser.add_argument("folder", nargs="?", help="Thư mục nhóm OCR; mặc định user_003")
    args = parser.parse_args()
    main(args.folder)
