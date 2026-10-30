"""
scripts/make_synthetic_data.py
------------------------------
Generate a SMALL SYNTHETIC leaf dataset so the whole pipeline can be tested
in a couple of minutes before the real PlantVillage data is downloaded.

The images are cartoons (an ellipse-shaped green leaf on a grey background
with drawn "disease" patterns), NOT real data: never report results on them
in the paper. They only prove that every script runs end to end.

Usage (from the project root):
    python scripts/make_synthetic_data.py                 # -> data/synthetic, 60 images/class
    python scripts/make_synthetic_data.py --n 100 --out data/synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZE = 160
CLASSES = ["Tomato___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
           "Tomato___Late_blight", "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___mosaic_virus"]


def jitter(rng, rgb, amount=18):
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in rgb)


def base_leaf(rng):
    """Grey background with a random green elliptical leaf and a midrib."""
    bg = jitter(rng, (185, 185, 180), 25)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    w, h = rng.integers(55, 70), rng.integers(30, 42)
    cx, cy = SIZE // 2 + rng.integers(-8, 9), SIZE // 2 + rng.integers(-8, 9)
    green = jitter(rng, (60, 135, 45), 20)
    d.ellipse([cx - w, cy - h, cx + w, cy + h], fill=green)
    d.line([cx - w + 5, cy, cx + w - 5, cy], fill=jitter(rng, (95, 160, 75), 10), width=2)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).ellipse([cx - w, cy - h, cx + w, cy + h], fill=255)
    return img, mask, (cx, cy, w, h)


def random_point_in_leaf(rng, geom):
    cx, cy, w, h = geom
    while True:
        x, y = rng.uniform(-1, 1, 2)
        if x * x + y * y < 0.8:
            return int(cx + x * w), int(cy + y * h)


def draw_disease(img, mask, geom, cls, rng):
    d = ImageDraw.Draw(img)
    if cls.endswith("Bacterial_spot"):           # many small dark spots
        for _ in range(rng.integers(25, 45)):
            x, y = random_point_in_leaf(rng, geom)
            r = rng.integers(1, 4)
            d.ellipse([x - r, y - r, x + r, y + r], fill=jitter(rng, (55, 40, 25), 12))
    elif cls.endswith("Early_blight"):           # few large ringed spots with yellow halo
        for _ in range(rng.integers(3, 6)):
            x, y = random_point_in_leaf(rng, geom)
            r = rng.integers(7, 12)
            d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=jitter(rng, (190, 180, 60), 15))
            d.ellipse([x - r, y - r, x + r, y + r], fill=jitter(rng, (115, 75, 40), 12))
            d.ellipse([x - r // 2, y - r // 2, x + r // 2, y + r // 2], outline=(70, 45, 25), width=1)
    elif cls.endswith("Late_blight"):            # big dark irregular water-soaked patches
        for _ in range(rng.integers(2, 4)):
            x, y = random_point_in_leaf(rng, geom)
            pts = [(x + rng.integers(-22, 23), y + rng.integers(-15, 16)) for _ in range(7)]
            d.polygon(pts, fill=jitter(rng, (60, 65, 50), 10))
    elif cls.endswith("Yellow_Leaf_Curl_Virus"):  # yellowing from the edges
        cx, cy, w, h = geom
        yellow = Image.new("RGB", img.size, jitter(rng, (185, 175, 55), 15))
        inner = Image.new("L", img.size, 0)
        s = rng.uniform(0.45, 0.65)
        ImageDraw.Draw(inner).ellipse([cx - w * s, cy - h * s, cx + w * s, cy + h * s], fill=255)
        edge = Image.composite(Image.new("L", img.size, 0), mask, inner).filter(ImageFilter.GaussianBlur(4))
        img.paste(yellow, (0, 0), edge)
    elif cls.endswith("mosaic_virus"):           # light / dark green mottling
        for _ in range(rng.integers(12, 20)):
            x, y = random_point_in_leaf(rng, geom)
            r = rng.integers(4, 9)
            d.ellipse([x - r, y - r, x + r, y + r], fill=jitter(rng, (140, 190, 90), 12))
    return img


def make_image(cls, rng):
    img, mask, geom = base_leaf(rng)
    img = draw_disease(img, mask, geom, cls, rng)
    img = img.rotate(rng.uniform(0, 360), resample=Image.BILINEAR, fillcolor=img.getpixel((2, 2)))
    arr = np.asarray(img).astype(float) + rng.normal(0, 6, (SIZE, SIZE, 3))   # camera noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=60, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for cls in CLASSES:
        folder = Path(args.out) / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            make_image(cls, rng).save(folder / f"{i:04d}.jpg", quality=92)
        print(f"wrote {args.n} images to {folder}")


if __name__ == "__main__":
    main()
