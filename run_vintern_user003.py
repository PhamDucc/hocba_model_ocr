from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()

    scripts = [
        "test_vintern_groups.py",
        "extract_vintern_groups.py",
    ]

    for script in scripts:
        print(f"\nĐang chạy {script}...", flush=True)

        subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            check=True,
        )

    print(
        f"\nHoàn tất trong "
        f"{time.perf_counter() - started:.1f} giây."
    )
    print(
        "JSON:",
        ROOT / "results/vintern_groups/user_003/student.json",
    )


if __name__ == "__main__":
    main()