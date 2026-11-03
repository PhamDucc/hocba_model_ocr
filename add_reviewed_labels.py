from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def read_labels(path):
    result = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        if '\t' not in line:
            raise ValueError(f'{path}:{n}: Thiếu TAB giữa đường dẫn và nhãn')
        name, text = line.split('\t', 1)
        name, text = name.strip().replace('\\', '/'), text.strip()
        if not text or not (DATA / name).is_file():
            raise ValueError(f'{path}:{n}: Nhãn trống hoặc thiếu ảnh: {name}')
        if name in result and result[name] != text:
            raise ValueError(f'Nhãn mâu thuẫn: {name}')
        result[name] = text
    return result


def main():
    updates = []
    # Validate both sets before writing either file.
    for split in ('train', 'val'):
        target = DATA / f'{split}_labels.txt'
        old = read_labels(target)
        extra = read_labels(DATA / f'{split}_labels_real_add.txt')
        for name, text in extra.items():
            if not name.startswith(split + '/'):
                raise ValueError(f'Sai tập: {name}')
            if name in old and old[name] != text:
                raise ValueError(f'Nhãn hiện tại khác nhãn thêm: {name}')
        merged = dict(old)
        merged.update(extra)
        updates.append((target, old, merged))
    train_names = set(updates[0][2])
    val_names = set(updates[1][2])
    if train_names & val_names:
        raise ValueError('Một đường dẫn ảnh xuất hiện trong cả train và val')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for target, old, merged in updates:
        added = len(merged) - len(old)
        if added:
            backup = target.with_name(target.name + '.' + stamp + '.bak')
            shutil.copy2(target, backup)
            temporary = target.with_suffix('.tmp')
            temporary.write_text(''.join(f'{p}\t{t}\n' for p, t in merged.items()), encoding='utf-8')
            temporary.replace(target)
            print('Sao lưu:', backup)
        print(f'{target.name}: {len(merged)} ảnh, thêm {added}')
    print('Đã cập nhật nhãn. Chưa chạy huấn luyện.')


if __name__ == '__main__':
    main()
