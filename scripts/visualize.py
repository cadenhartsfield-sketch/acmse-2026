"""
visualize.py  –  Side-by-side visualization of original image vs segmentation mask.

Usage:
    python scripts/visualize.py                          # visualize all test images
    python scripts/visualize.py --file 108082.png        # single image
    python scripts/visualize.py --images train/images --masks train/masks  # train set
    python scripts/visualize.py --count 6               # first N images only

Output:
    visualizations/<filename>  –  side-by-side PNG (original | colorized segments)
"""

import os
import argparse
import numpy as np
from PIL import Image


# ─── deterministic colour palette ────────────────────────────────────────────
def _make_palette(n: int, seed: int = 42) -> np.ndarray:
    """Return (n, 3) uint8 RGB palette, reproducible across runs."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 230, size=(max(n, 1), 3), dtype=np.uint8)
    return palette


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map integer segment labels → RGB image."""
    n_segs  = int(mask.max()) + 1
    palette = _make_palette(n_segs)
    rgb     = palette[mask % len(palette)]   # wrap for safety
    return rgb.astype(np.uint8)


def side_by_side(gray: np.ndarray, colored: np.ndarray) -> Image.Image:
    """Stitch grayscale (as RGB) and coloured mask side by side."""
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    canvas   = np.concatenate([gray_rgb, colored], axis=1)
    return Image.fromarray(canvas)


def process(img_path: str, mask_path: str, out_path: str) -> int:
    gray = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path), dtype=np.int32)
    colored  = colorize(mask)
    combined = side_by_side(gray, colored)
    combined.save(out_path)
    return int(mask.max()) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="test/images")
    parser.add_argument("--masks",  default="predicted_masks")
    parser.add_argument("--out",    default="visualizations")
    parser.add_argument("--file",   default=None, help="Process a single filename")
    parser.add_argument("--count",  type=int, default=None, help="Limit to first N images")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.file:
        files = [args.file]
    else:
        files = sorted(f for f in os.listdir(args.masks) if f.endswith(".png"))
        if args.count:
            files = files[:args.count]

    print(f"Visualizing {len(files)} image(s) → '{args.out}/'")

    for fname in files:
        img_path  = os.path.join(args.images, fname)
        mask_path = os.path.join(args.masks,  fname)
        out_path  = os.path.join(args.out,    fname)

        if not os.path.exists(img_path):
            print(f"  [SKIP] {fname} – image not found")
            continue
        if not os.path.exists(mask_path):
            print(f"  [SKIP] {fname} – mask not found")
            continue

        n = process(img_path, mask_path, out_path)
        print(f"  {fname}  ({n:,} segments)  →  {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
