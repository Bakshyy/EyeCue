# src/training/build_labels_csv_from_yolo.py
#
# Build labels.csv from YOLO-format labels under data/processed/<split>.
# For each image:
#   - If there is a YOLO label with class 0 and non-zero box -> has_ball = 1.
#   - Otherwise -> has_ball = 0 and bbox = (0, 0, 0, 0).

from pathlib import Path
import csv

BASE_DIR = Path("data") / "processed"
SPLITS = ["train", "val", "test"]

# Allowed image extensions
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_yolo_label(label_path: Path):
    """
    Parse a YOLO label file.

    Returns:
        (cx, cy, w, h, has_ball)

        - If a valid class-0 box is found: normalized coords in [0,1], has_ball=1.
        - If no valid box: (0,0,0,0,0).
    """
    if not label_path.exists():
        # No label file at all -> treat as no ball
        return 0.0, 0.0, 0.0, 0.0, 0

    text = label_path.read_text().strip()
    if not text:
        # Empty file -> no ball
        return 0.0, 0.0, 0.0, 0.0, 0

    cx = cy = w = h = 0.0
    has_ball = 0

    # Each non-empty line should be: <class_id> cx cy w h
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue

        # Only treat class 0 with positive size as "ball present"
        if cls == 0 and bw > 0.0 and bh > 0.0:
            cx, cy, w, h = x, y, bw, bh
            has_ball = 1
            # Use the first valid ball box and ignore any others
            break

    if has_ball == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    else:
        # Clamp just in case YOLO produced slightly out-of-range numbers
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        return cx, cy, w, h, 1


def process_split(split: str):
    label_dir = BASE_DIR / split / "labels"
    img_dir = BASE_DIR / split / "images"

    print(f"\n[INFO] Processing split '{split}'")
    print(f"  Label dir : {label_dir}")
    print(f"  Images dir: {img_dir}")

    if not img_dir.exists():
        print(f"[WARN] Images dir does not exist for split '{split}': {img_dir}")
        return

    label_dir.mkdir(parents=True, exist_ok=True)
    csv_path = label_dir / "labels.csv"

    rows = []
    total_images = 0
    no_label_file = 0
    empty_or_invalid_label = 0
    has_ball_count = 0
    no_ball_count = 0

    # Iterate over IMAGES, not labels
    for img_path in sorted(img_dir.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in EXTS:
            continue

        total_images += 1
        stem = img_path.stem
        lbl_path = label_dir / f"{stem}.txt"

        if not lbl_path.exists():
            no_label_file += 1

        cx, cy, w, h, has_ball = parse_yolo_label(lbl_path)

        if has_ball == 1:
            has_ball_count += 1
        else:
            no_ball_count += 1
            # Count as "empty/invalid" if label file exists but had no usable box
            if lbl_path.exists():
                empty_or_invalid_label += 1

        rows.append(
            {
                "filename": img_path.name,
                "cx": f"{cx:.6f}",
                "cy": f"{cy:.6f}",
                "w": f"{w:.6f}",
                "h": f"{h:.6f}",
                "has_ball": has_ball,
            }
        )

    # Write CSV
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "cx", "cy", "w", "h", "has_ball"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[INFO] Split '{split}' summary:")
    print(f"  Total images           : {total_images}")
    print(f"  Images w/ NO label file: {no_label_file}")
    print(f"  Label files w/o ball   : {empty_or_invalid_label}")
    print(f"  Rows written to CSV    : {len(rows)}")
    print(f"  has_ball = 1 rows      : {has_ball_count}")
    print(f"  has_ball = 0 rows      : {no_ball_count}")
    print(f"  CSV path               : {csv_path}")


def main():
    print("[INFO] Building labels.csv from YOLO .txt files (using data/processed)...")
    for split in SPLITS:
        process_split(split)
    print("[INFO] Done building CSVs.")


if __name__ == "__main__":
    main()
