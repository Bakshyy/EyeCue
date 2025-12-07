# src/training/build_labels_csv.py

import csv
from pathlib import Path
from typing import List, Optional, Tuple


# --- CONFIG ---

# YOLO class id that corresponds to "tennis ball"
TENnis_BALL_CLASS_ID = 0

# Valid image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_project_root() -> Path:
    """
    Return the project root directory.

    This assumes the file is located at:
        <project_root>/src/training/build_labels_csv.py
    so project_root is two levels up from this file.
    """
    return Path(__file__).resolve().parents[2]


def parse_yolo_label_file(label_path: Path) -> Tuple[bool, float, float, float, float]:
    """
    Parse a YOLO-format label file:
        class_id cx cy w h  (all floats, normalized 0..1)

    Returns:
        has_ball (bool),
        cx, cy, w, h (floats in [0,1], or 0 if no tennis-ball object)

    Rules:
      - If file doesn't exist or is empty → has_ball = False, bbox = 0,0,0,0
      - If multiple tennis-ball objects (class_id == TENnis_BALL_CLASS_ID),
        we take the FIRST one.
      - Lines with other class_ids are ignored.
    """
    if not label_path.exists():
        # No label file → treat as no ball
        return False, 0.0, 0.0, 0.0, 0.0

    content = label_path.read_text().strip()
    if not content:
        # Empty file → no ball
        return False, 0.0, 0.0, 0.0, 0.0

    has_ball = False
    cx = cy = w = h = 0.0

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            # Malformed line, skip but warn
            print(f"[WARN] Malformed label line in {label_path}: '{line}'")
            continue

        try:
            class_id = int(float(parts[0]))
            cx_tmp = float(parts[1])
            cy_tmp = float(parts[2])
            w_tmp = float(parts[3])
            h_tmp = float(parts[4])
        except ValueError:
            print(f"[WARN] Could not parse numbers in label file {label_path}: '{line}'")
            continue

        if class_id == TENnis_BALL_CLASS_ID:
            # Take the first tennis-ball object and ignore others
            has_ball = True
            cx, cy, w, h = cx_tmp, cy_tmp, w_tmp, h_tmp
            break

    if not has_ball:
        return False, 0.0, 0.0, 0.0, 0.0

    # Basic sanity clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return has_ball, cx, cy, w, h


def build_labels_for_split(
    images_dir: Path,
    labels_dir: Path,
    split: str,
) -> None:
    """
    Build labels.csv for the given split ("train", "val", "test").

    Writes:
        labels_dir / "labels.csv"

    CSV columns:
        filename,cx,cy,w,h,has_ball
    """
    split_images_dir = images_dir / split
    split_labels_dir = labels_dir / split
    csv_path = split_labels_dir / "labels.csv"

    if not split_images_dir.exists():
        print(f"[WARN] Images dir for split '{split}' does not exist: {split_images_dir}")
        return

    split_labels_dir.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = []

    num_images = 0
    num_with_label_file = 0
    num_has_ball = 0

    # Sort for deterministic ordering
    for img_path in sorted(split_images_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        num_images += 1
        stem = img_path.stem
        label_path = split_labels_dir / f"{stem}.txt"

        if label_path.exists():
            num_with_label_file += 1

        has_ball, cx, cy, w, h = parse_yolo_label_file(label_path)
        if has_ball:
            num_has_ball += 1

        row = [
            img_path.name,        # filename
            f"{cx:.6f}",          # cx
            f"{cy:.6f}",          # cy
            f"{w:.6f}",           # w
            f"{h:.6f}",           # h
            "1" if has_ball else "0",  # has_ball
        ]
        rows.append(row)

    # Write CSV
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cx", "cy", "w", "h", "has_ball"])
        writer.writerows(rows)

    print(f"\n[INFO] Built labels CSV for split '{split}': {csv_path}")
    print(f"  Total images:            {num_images}")
    print(f"  With .txt label file:    {num_with_label_file}")
    print(f"  Images with tennis ball: {num_has_ball}")
    print(f"  Images without ball:     {num_images - num_has_ball}")


def main():
    project_root = find_project_root()
    data_dir = project_root / "data"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Images root:  {images_dir}")
    print(f"[INFO] Labels root:  {labels_dir}")

    for split in ["train", "val", "test"]:
        build_labels_for_split(images_dir, labels_dir, split)

    print("\n[INFO] Done building labels.csv files for all splits.")


if __name__ == "__main__":
    main()
