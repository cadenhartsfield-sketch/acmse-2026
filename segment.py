import os
from py_compile import main
import numpy as np
from PIL import Image
from collections import deque


def segment_image(img_array, threshold=15):
    h, w = img_array.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    seg_id = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1:
                continue
            queue = deque()
            queue.append((r, c))
            labels[r, c] = seg_id
            while queue:
                pr, pc = queue.popleft()
                pval = int(img_array[pr, pc])
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                        if abs(int(img_array[nr, nc]) - pval) <= threshold:
                            labels[nr, nc] = seg_id
                            queue.append((nr, nc))
                seg_id += 1
        return labels


def main():
    input_dir = "test/images"
    output_dir = "predicted_masks"
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".png"))
    for i, fname in enumerate(files, 1):
        img = np.array(Image.open(os.path.join(input_dir, fname)).convert("L"))
        labels = segment_image(img)
        out = Image.fromarray(labels.astype(np.int32), mode="I")
        out.save(os.path.join(output_dir, fname))
        print(f"[{i}/{len(files)}] {fname} - {labels.max()+1} segments")


if __name__ == "__main__":
    main()
