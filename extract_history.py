from pathlib import Path
import json
import unicodedata

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "results/full_page/synthetic_010"


def normalize(text):
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return " ".join(
        "".join(
            c for c in text
            if unicodedata.category(c) != "Mn"
        ).split()
    )


def bounds(region):
    points = region["polygon"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def center(region):
    x1, y1, x2, y2 = bounds(region)
    return (x1 + x2) / 2, (y1 + y2) / 2


def main():
    source = json.loads(
        (FOLDER / "ocr.json").read_text(encoding="utf-8")
    )
    regions = source["regions"]

    title = next(
        (
            r for r in regions
            if "qua trinh hoc tap" in normalize(r["text"])
        ),
        None,
    )
    if title is None:
        raise ValueError("Không tìm được tiêu đề quá trình học tập.")

    table_top = bounds(title)[3]

    header_names = {
        "nam hoc": "nam_hoc",
        "lop": "lop",
        "ten truong": "truong",
        "vao so dang bo": "so_dang_bo",
    }

    headers = {}

    for region in sorted(regions, key=lambda r: center(r)[1]):
        if center(region)[1] <= table_top:
            continue

            # Xử lý các vùng văn bản khác.
        label = normalize(region["text"])
        key = header_names.get(label)

        # Một số mẫu có tiêu đề cột dài hơn.
        if key is None and label.startswith("ten truong"):
            key = "truong"
        if key and key not in headers:
            headers[key] = region

    if len(headers) != 4:
        raise ValueError(
            "Chưa tìm đủ 4 tiêu đề cột. Tìm được: "
            + ", ".join(headers)
        )

    header_bottom = max(
        bounds(region)[3] for region in headers.values()
    )

    # Cần điểm kết thúc để tránh lấy chữ ký vào bảng.
    footers = [
        r for r in regions
        if center(r)[1] > header_bottom
        and (
            "hieu truong" in normalize(r["text"])
            or (
                "ngay" in normalize(r["text"])
                and "thang" in normalize(r["text"])
                and "nam" in normalize(r["text"])
            )
        )
    ]

    if not footers:
        raise ValueError(
            "Chưa xác định được cuối bảng; cần kiểm tra ocr.json."
        )

    table_bottom = min(bounds(r)[1] for r in footers)

    columns = sorted(
        headers,
        key=lambda key: center(headers[key])[0],
    )
    column_centers = [
        center(headers[key])[0] for key in columns
    ]
    boundaries = [
        (a + b) / 2
        for a, b in zip(column_centers, column_centers[1:])
    ]

    candidates = [
        r for r in regions
        if header_bottom < center(r)[1] < table_bottom
    ]

    rows = []

    for region in sorted(candidates, key=lambda r: center(r)[1]):
        _, y1, _, y2 = bounds(region)
        cy = center(region)[1]
        height = max(1, y2 - y1)

        if rows and abs(cy - rows[-1]["cy"]) <= (
            0.5 * min(height, rows[-1]["height"])
        ):
            rows[-1]["regions"].append(region)
            rows[-1]["cy"] = sum(
                center(r)[1] for r in rows[-1]["regions"]
            ) / len(rows[-1]["regions"])
        else:
            rows.append({
                "cy": cy,
                "height": height,
                "regions": [region],
            })

    history = []

    for row in rows:
        cells = {key: [] for key in columns}

        for region in sorted(
            row["regions"], key=lambda r: center(r)[0]
        ):
            cx = center(region)[0]
            column_index = sum(cx > b for b in boundaries)
            cells[columns[column_index]].append(region)

        values = {
            key: " ".join(
                r["text"].strip() for r in cells[key]
            ) or None
            for key in columns
        }

        history.append({
            **values,
            "needs_review": True,
            "evidence": cells,
        })

    result = {
        "source": source["source"],
        "qua_trinh_hoc_tap": history,
        "note": (
            "Ghép theo vị trí hàng/cột, giữ nguyên kết quả OCR. "
            "Cần kiểm tra lại, nhất là bảng nghiêng hoặc khác bố cục."
        ),
    }

    destination = FOLDER / "history.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Nếu đã có student.json, bổ sung bảng vào file đó.
    student_path = FOLDER / "student.json"
    if student_path.exists():
        student = json.loads(
            student_path.read_text(encoding="utf-8")
        )
        student["qua_trinh_hoc_tap"] = history
        student.pop("history_error", None)
        student_path.write_text(
            json.dumps(student, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for index, item in enumerate(history, 1):
        print(f"\nHàng {index}:")
        for key in columns:
            print(f"  {key}: {item[key]}")

    print("\nĐã lưu:", destination)


if __name__ == "__main__":
    main()