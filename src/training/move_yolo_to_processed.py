# src/training/move_yolo_to_processed.py
"""
Moves YOLO auto-labeled dataset from:
    data/images/all
    data/labels/all

into the structure expected by split_dataset.py:
    data/processed/all/images
    data/processed/all/labels
"""

from pathlib import Path
import shutil

SRC_IMAGES = Path("data/images/all")
SRC_LABELS = Path("data/labels/all")

DEST_IMAGES = Path("data/processed/all/images")
DEST_LABELS = Path("data/processed/all/labels")


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        print(f"[WARN] Source path does not exist: {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in src.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            count += 1
    print(f"[INFO] Copied {count} files → {dst}")


def main():
    print("[INFO] Moving YOLO dataset to data/processed/all/...")

    # Ensure the processed directories exist
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)

    # Copy files
    copy_tree(SRC_IMAGES, DEST_IMAGES)
    copy_tree(SRC_LABELS, DEST_LABELS)

    print("[INFO] Done moving dataset.")


if __name__ == "__main__":
    main()
