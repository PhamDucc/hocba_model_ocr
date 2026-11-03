from pathlib import Path
import argparse
import subprocess
import sys
import time

import extract_vintern_groups as extractor


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    source = Path(args.image)
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    started = time.perf_counter()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "test_vintern_groups.py"),
            str(source),
        ],
        cwd=ROOT,
        check=True,
    )

    extractor.FOLDER = (
        ROOT / "results/vintern_groups" / source.stem
    )
    extractor.main()

    print(
        f"\nHoàn tất: "
        f"{time.perf_counter() - started:.1f} giây"
    )


if __name__ == "__main__":
    main()