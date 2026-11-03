from pathlib import Path
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "results/full_page/synthetic_010"

LABELS = {
    "ho va ten nguoi giam ho": "nguoi_giam_ho",
    "nghe nghiep nguoi giam ho": "nghe_nguoi_giam_ho",
    "ho va ten cha": "ho_ten_cha",
    "ho va ten me": "ho_ten_me",
    "cho o hien tai": "dia_chi",
    "ho va ten": "ho_ten",
    "gioi tinh": "gioi_tinh",
    "ngay sinh": "ngay_sinh",
    "noi sinh": "noi_sinh",
    "dan toc": "dan_toc",
    "doi tuong": "doi_tuong",
    "nghe nghiep": "nghe_nghiep",
}


def fold(text):
    # Giữ vị trí ký tự để cắt giá trị từ văn bản gốc.
    text = unicodedata.normalize("NFC", text)
    result = []

    for char in text:
        char = char.lower().replace("đ", "d")
        result.append("".join(
            c for c in unicodedata.normalize("NFD", char)
            if unicodedata.category(c) != "Mn"
        ))

    return "".join(result)


def parse_label(text):
    text = unicodedata.normalize("NFC", text).strip()
    normalized = fold(text)

    for label, key in LABELS.items():
        pattern = (
            r"^\s*"
            + r"\s+".join(re.escape(word) for word in label.split())
            + r"(?=$|[\s:.\-–—])"
        )
        match = re.match(pattern, normalized)

        if match:
            value = text[match.end():].lstrip(
                " \t:.-–—"
            ).strip()
            return key, value or None

    return None, None


def bounds(region):
    points = region["polygon"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def same_row(left, right):
    _, y1, _, y2 = bounds(left)
    _, ry1, _, ry2 = bounds(right)

    h = max(1, y2 - y1)
    rh = max(1, ry2 - ry1)
    overlap = max(0, min(y2, ry2) - max(y1, ry1))
    center_gap = abs((y1 + y2 - ry1 - ry2) / 2)

    return (
        overlap / min(h, rh) >= 0.3
        and center_gap <= 0.6 * max(h, rh)
    )


def find_value(label_region, regions, consumed):
    x1, y1, x2, y2 = bounds(label_region)
    height = max(1, y2 - y1)
    label_center_x = (x1 + x2) / 2
    label_center_y = (y1 + y2) / 2
    candidates = []

    for other in regions:
        if other["index"] == label_region["index"]:
            continue
        if other["index"] in consumed:
            continue
        if not other["text"].strip():
            continue
        if parse_label(other["text"])[0] is not None:
            continue

        ox1, oy1, ox2, oy2 = bounds(other)
        gap = ox1 - x2
        center_x = (ox1 + ox2) / 2
        center_y = (oy1 + oy2) / 2

        # Cho phép hai khung hơi chồng nhau,
        # nhưng giá trị vẫn phải nằm bên phải và cùng hàng.
        if (
            -0.5 * height <= gap <= 12 * height
            and center_x > label_center_x
            and same_row(label_region, other)
        ):
            candidates.append((
                abs(center_y - label_center_y),
                abs(gap),
                other,
            ))

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )[2]

def validate_birth_date(value):
    from datetime import date

    if not value:
        return {
            "status": "missing",
            "iso": None,
            "message": "Chưa trích xuất được ngày sinh.",
        }

    text = " ".join(fold(value).split())
    match = re.fullmatch(
        r"(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})",
        text,
    )

    if not match:
        return {
            "status": "needs_review",
            "iso": None,
            "message": "Thiếu thành phần hoặc chưa nhận diện được định dạng.",
        }

    day, month, year = map(int, match.groups())

    try:
        birthday = date(year, month, day)
    except ValueError:
        return {
            "status": "needs_review",
            "iso": None,
            "message": "Ngày, tháng hoặc năm không hợp lệ.",
        }

    if birthday > date.today():
        return {
            "status": "needs_review",
            "iso": None,
            "message": "Ngày sinh nằm trong tương lai.",
        }

    return {
        "status": "format_valid",
        "iso": birthday.isoformat(),
        "message": (
            "Đúng định dạng và hợp lệ theo lịch; "
            "chưa xác nhận khớp với ảnh gốc."
        ),
    }

def main():
    source = json.loads(
        (FOLDER / "ocr.json").read_text(encoding="utf-8")
    )

    # Chỉ lấy vùng thông tin nằm phía trên tiêu đề bảng.
    titles = [
        r for r in source["regions"]
        if "qua trinh hoc tap" in fold(r["text"])
    ]
    if not titles:
        raise ValueError(
            "Không xác định được ranh giới phần thông tin cá nhân."
        )

    cutoff = min(bounds(r)[1] for r in titles)

    regions = sorted(
        [
            r for r in source["regions"]
            if bounds(r)[3] <= cutoff
        ],
        key=lambda r: (
            (bounds(r)[1] + bounds(r)[3]) / 2,
            bounds(r)[0],
        ),
    )

    fields = {
        key: None for key in LABELS.values()
        if key not in {"nghe_nghiep", "nghe_nguoi_giam_ho"}
    }
    fields["nghe_nghiep_cha"] = None
    fields["nghe_nghiep_me"] = None

    evidence = {}
    consumed = set()
    occupations = []

    for region in regions:
        if region["index"] in consumed:
            continue

        key, value = parse_label(region["text"])
        if key is None or key == "nghe_nguoi_giam_ho":
            continue

        parts = [region]

        if value is None:
            candidate = find_value(region, regions, consumed)
            if candidate:
                value = candidate["text"].strip()
                parts.append(candidate)
                consumed.add(candidate["index"])

        item = {
            "text": region["text"],
            "parts": parts,
        }

        if key == "nghe_nghiep":
            occupations.append((region, value, item))
        elif key in fields:
            fields[key] = value
            evidence[key] = item

    # Ghép nghề nghiệp theo cùng hàng với cha/mẹ,
    # thay vì phụ thuộc thứ tự xuất hiện trong OCR.
    used_occupations = set()

    for parent in ["cha", "me"]:
        parent_key = f"ho_ten_{parent}"
        if parent_key not in evidence:
            continue

        parent_region = evidence[parent_key]["parts"][0]
        candidates = []

        for i, (region, value, item) in enumerate(occupations):
            if i in used_occupations:
                continue

            if (
                bounds(region)[0] >= bounds(parent_region)[2]
                and same_row(parent_region, region)
            ):
                gap = bounds(region)[0] - bounds(parent_region)[2]
                candidates.append((gap, i, value, item))

        if candidates:
            _, i, value, item = min(
                candidates, key=lambda candidate: candidate[0]
            )
            key = f"nghe_nghiep_{parent}"
            fields[key] = value
            evidence[key] = item
            used_occupations.add(i)

    destination = FOLDER / "student.json"
    result = {}

    # Giữ bảng học tập đã trích xuất cho cùng ảnh.
    if destination.exists():
        old = json.loads(destination.read_text(encoding="utf-8"))
        if old.get("source") == source["source"]:
            for key in ["qua_trinh_hoc_tap", "history_error"]:
                if key in old:
                    result[key] = old[key]

    result.update({
        "source": source["source"],
        "fields": fields,
        "evidence": evidence,
        "note": (
            "Giữ nguyên giá trị OCR, chưa xác minh. "
            "Nhãn và giá trị được ghép theo tọa độ; "
            "null có thể là ô trống hoặc chưa trích xuất được."
        ),
    })

    result["validation"] = {
        "ngay_sinh": validate_birth_date(
            fields.get("ngay_sinh")
        )
    }

    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(fields, ensure_ascii=False, indent=2))
    print("\nĐã lưu:", destination)


if __name__ == "__main__":
    main()