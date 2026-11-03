from pathlib import Path
from PIL import Image

folder = Path("results/full_page/user_003/crops")

with Image.open(folder / "line_015.png") as image:
    w, h = image.size
    image.crop((int(w * 0.45), 0, w, h)).save(
        folder / "dia_chi_tail.png"
    )
    