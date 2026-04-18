# ACMSE 2026 – Image Segmentation

## What It Does

Takes grayscale images and groups pixels into segments based on intensity similarity. Two pixels end up in the same segment if they're directly touching (up, down, left, right) and their brightness values are within 15 of each other.

## How It Works

I used BFS (Breadth-First Search) for the flood fill. I went with BFS over other approaches because it explores level by level, which felt like the right fit for spreading outward from a pixel to its neighbors. Each unvisited pixel starts a new segment, and BFS handles the rest.

## Problems I Ran Into

Getting the indentation right in Python was the biggest hurdle — the nested loops inside the BFS had a lot of levels to keep track of. I also had to switch from 8-bit to 32-bit PNG output because some images produced way more than 255 segments.

## How to Run It

```
python3 segment.py
```

Output masks are saved to `predicted_masks/`.

## Dependencies

```
pip3 install numpy pillow
```

## Why This Works

### Zero Rule Violations

After generating all 50 masks, I added a validation script (`verify.py`) which checked all adjacent pairs across all output images. The rule states that two pixels belong to the same segment if they are adjacent (up, down, left, right), and their intensity difference is less than or equal to 15. Differences of 16 or more cause a split. The result of this was zero violations throughout all 50 images, confirming every mask follows the contest rules.

### 32-Bit PNG Encoding

An 8-bit PNG can only store values from 0 to 255, which means that the cap is 256 unique segment IDs. Some images within this dataset produce over 153,000 segments at threshold=15. Therefore, saving this as an 8-bit would wrap the segment IDs back to 0. Because of this, I switched to 32-bit integer PNG output.
