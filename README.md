# ACMSE 2026 – Image Segmentation

## What it does

Takes grayscale images and groups pixels into segments based on
intensity similarity. Two pixels end up in the same segment if
they're directly touching (up, down, left, right) and their
brightness values are within 15 of each other.

## How it works

I used BFS (Breadth-First Search) for the flood fill. I went with
BFS over other approaches because it explores level by level, which
felt like the right fit for spreading outward from a pixel to its
neighbors. Each unvisited pixel starts a new segment, and BFS
handles the rest.

## Problems I ran into

Getting the indentation right in Python was the biggest hurdle —
the nested loops inside the BFS had a lot of levels to keep track
of. I also had to switch from 8-bit to 32-bit PNG output because some
images produced way more than 255 segments.

## How to run it

python3 segment.py

Output masks are saved to predicted_masks/

## Dependencies

pip3 install numpy pillow
