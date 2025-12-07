# src/pc/auto_label_with_yolo.py
"""
Run YOLO on raw images and build a YOLO-style dataset under:
  data/images/all
  data/labels/all

Rules:
  - Every image gets a .txt label file.
  - Ball subsets (self_ball, supp_ball) use YOLO boxes when present.
  - Background subsets force empty label files (negative examples).
"""

from pathlib import Path
import argparse
import shutil

from ultralytics import YOLO

# Raw data layout
RAW_BASE = Path("data") / "raw"

SUBSETS = [
    # (subdir name, expect_ball, conf_threshold)
    ("self_ball", True, 0.10),
    ("supp_ball", True, 0.25),
    ("self_background", False, 0.25),
    ("supp_background", False, 0.25),
]

# Where we build the unified YOLO dataset
IMAGES_ALL_DIR = Path("data") / "images" / "all"
LABELS_ALL_DIR = Path("data") / "labels" / "all"

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(root: Path):
    for p in sorted(root.glob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def write_label(label_path: Path, boxes_norm):
    """
    boxes_norm: list of (cx, cy, w, h) in normalized [0,1] coords.
    Always creates a label file (empty file => no ball).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for cx, cy, w, h in boxes_norm:
            # single class "ball" -> class_id 0
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolo_ball.pt",
        help="Path to YOLO weights (.pt)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    print(f"[INFO] Loading YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))

    IMAGES_ALL_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_ALL_DIR.mkdir(parents=True, exist_ok=True)

    for subset_name, expect_ball, conf_th in SUBSETS:
        subset_dir = RAW_BASE / subset_name
        if not subset_dir.exists():
            print(f"[WARN] Raw subset dir does not exist, skipping: {subset_dir}")
            continue

        img_paths = list(iter_images(subset_dir))
        total = len(img_paths)

        print(
            f"\n[INFO] Processing '{subset_name}' "
            f"(expect_ball={expect_ball}, conf={conf_th})"
        )
        print(f"  Found {total} images")

        det_count = 0
        no_det_count = 0

        for img_path in img_paths:
            # Copy image into data/images/all
            dst_img = IMAGES_ALL_DIR / img_path.name
            shutil.copy2(img_path, dst_img)

            label_path = LABELS_ALL_DIR / (img_path.stem + ".txt")

            boxes_norm = []

            if expect_ball:
                # Run YOLO
                results = model(str(img_path), conf=conf_th, verbose=False)
                r = results[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    # Assume our model has ball as class 0 and we just take all boxes
                    xywhn = r.boxes.xywhn.cpu().numpy()  # (N, 4)
                    for cx, cy, w, h in xywhn:
                        boxes_norm.append(
                            (float(cx), float(cy), float(w), float(h))
                        )
                    det_count += 1
                else:
                    # Expected a ball but YOLO missed it -> treat as "no ball" for now
                    no_det_count += 1
            else:
                # Background subsets: ALWAYS negative
                no_det_count += 1

            # Always write a label file, even if empty (=> negative)
            write_label(label_path, boxes_norm)

        print(f"[STATS] {subset_name}:")
        print(f"  total images     : {total}")
        print(f"  with ball boxes  : {det_count}")
        print(f"  empty label file : {no_det_count}")

    print(
        "\n[INFO] Done. YOLO labels written to", LABELS_ALL_DIR,
        "and images copied to", IMAGES_ALL_DIR,
    )


if __name__ == "__main__":
    main()

