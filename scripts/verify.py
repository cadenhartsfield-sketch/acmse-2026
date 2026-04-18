"""
verify.py  –  Validate segmentation masks against the stated contest rules.

Usage:
    python scripts/verify.py                         # verifies predicted_masks/ vs test/images/
    python scripts/verify.py --masks predicted_masks --images test/images
    python scripts/verify.py --against-gt            # also compares to train ground truth
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_mask(path):
    img = Image.open(path)
    return np.array(img, dtype=np.int32)


def verify_mask(img: np.ndarray, mask: np.ndarray, fname: str) -> dict:
    """
    Check two rules:
      1. COMPLETENESS – every pixel is assigned (no -1).
      2. RULE COMPLIANCE – no adjacent pair with diff<=15 is split across segments.

    Note: same-segment pairs CAN have diff>15 due to transitivity (BFS chains).
    That is expected and correct behaviour.
    """
    h, w = img.shape
    assert mask.shape == (h, w), f"{fname}: shape mismatch {mask.shape} vs {img.shape}"

    # Rule 1: completeness
    unassigned = int((mask < 0).sum())

    # Rule 2: no adjacent similar-pixel pair should be in different segments
    violations = 0
    for dr, dc, sl1, sl2 in [
        (1, 0, (slice(None,-1), slice(None)), (slice(1,None), slice(None))),
        (0, 1, (slice(None), slice(None,-1)), (slice(None), slice(1,None))),
    ]:
        diff = np.abs(img[sl1].astype(int) - img[sl2].astype(int))
        diff_seg = mask[sl1] != mask[sl2]
        violations += int(((diff <= 15) & diff_seg).sum())   # should be 0

    n_segs = int(mask.max()) + 1

    return {
        "file": fname,
        "segments": n_segs,
        "unassigned_pixels": unassigned,
        "rule_violations": violations,
        "ok": (unassigned == 0 and violations == 0),
    }


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of pixels where pred and GT agree (after remapping GT IDs)."""
    # remap GT to match pred via majority vote per GT segment
    gt_flat, pred_flat = gt.ravel(), pred.ravel()
    total = len(gt_flat)
    correct = 0
    for gt_id in np.unique(gt_flat):
        idx = gt_flat == gt_id
        if idx.sum() == 0:
            continue
        pred_ids, counts = np.unique(pred_flat[idx], return_counts=True)
        correct += counts.max()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks",   default="predicted_masks")
    parser.add_argument("--images",  default="test/images")
    parser.add_argument("--against-gt", action="store_true",
                        help="Also compute pixel accuracy vs train GT masks")
    args = parser.parse_args()

    mask_files = sorted(f for f in os.listdir(args.masks) if f.endswith(".png"))
    if not mask_files:
        print(f"No PNG masks found in '{args.masks}'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Verifying {len(mask_files)} masks")
    print(f"  Masks  : {args.masks}/")
    print(f"  Images : {args.images}/")
    print(f"{'='*60}\n")

    all_ok = True
    results = []

    for fname in mask_files:
        img_path  = os.path.join(args.images, fname)
        mask_path = os.path.join(args.masks,  fname)

        if not os.path.exists(img_path):
            print(f"  [SKIP] {fname} – no matching image found")
            continue

        img  = load_gray(img_path)
        mask = load_mask(mask_path)
        r    = verify_mask(img, mask, fname)
        results.append(r)

        status = "OK" if r["ok"] else "FAIL"
        print(f"  [{status:4s}]  {fname:<20s}  segs={r['segments']:6,d}  "
              f"unassigned={r['unassigned_pixels']}  violations={r['rule_violations']}")
        if not r["ok"]:
            all_ok = False

    total_segs = sum(r["segments"] for r in results)
    avg_segs   = total_segs / len(results) if results else 0

    print(f"\n{'─'*60}")
    print(f"  Images verified : {len(results)}")
    print(f"  All passed      : {all_ok}")
    print(f"  Avg segments    : {avg_segs:,.1f}")
    print(f"  Total segments  : {total_segs:,}")
    print(f"{'─'*60}\n")

    if args.against_gt:
        print("Ground-truth comparison (train set only) …")
        gt_dir  = "train/masks"
        img_dir = "train/images"
        accs = []
        for fname in sorted(os.listdir(gt_dir)):
            if not fname.endswith(".png"):
                continue
            pred_path = os.path.join("predicted_masks", fname)  # may not exist
            gt_path   = os.path.join(gt_dir,  fname)
            img_path  = os.path.join(img_dir, fname)
            if not os.path.exists(pred_path):
                continue  # run segment.py on train first
            pred = load_mask(pred_path)
            gt   = load_mask(gt_path)
            img  = load_gray(img_path)
            acc  = pixel_accuracy(pred, gt)
            accs.append(acc)
        if accs:
            print(f"  Mean pixel accuracy vs GT: {100*sum(accs)/len(accs):.2f}%")
        else:
            print("  No matching train masks found in predicted_masks/ (run segment.py on train/images first).")


if __name__ == "__main__":
    main()
