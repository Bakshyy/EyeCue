# src/training/split_dataset.py
# Split YOLO-labeled data/images/all + data/labels/all into
# data/processed/{train,val,test}/(images,labels).

import shutil
from pathlib import Path
import random

# Raw YOLO outputs
DATA_DIR = Path("data")
ALL_IMAGES_DIR = DATA_DIR / "images" / "all"
ALL_LABELS_DIR = DATA_DIR / "labels" / "all"

# Where we write the splits
PROCESSED_DIR = DATA_DIR / "processed"

SPLITS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def main():
    random.seed(42)

    # Collect all (image, label) pairs where the label file exists
    all_items = []
    for img_path in sorted(ALL_IMAGES_DIR.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in EXTS:
            continue

        stem = img_path.stem
        label_path = ALL_LABELS_DIR / f"{stem}.txt"
        if not label_path.exists():
            # If a label file doesn't exist at all, skip this image.
            # (auto_label_with_yolo should have created an empty .txt even for backgrounds;
            # if it didn’t, those images won’t be seen by the detector.)
            continue

        all_items.append((img_path, label_path))

    total = len(all_items)
    print(f"Total images with label files: {total}")

    if total == 0:
        print("No labeled images found in data/images/all + data/labels/all.")
        return

    random.shuffle(all_items)

    n_train = int(total * SPLITS["train"])
    n_val = int(total * SPLITS["val"])
    n_test = total - n_train - n_val

    print(f"Train: {n_train}")
    print(f"Val:   {n_val}")
    print(f"Test:  {n_test}")

    split_indices = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total),
    }

    # Clean + recreate output dirs
    for split in ["train", "val", "test"]:
        out_img_dir = PROCESSED_DIR / split / "images"
        out_lab_dir = PROCESSED_DIR / split / "labels"

        # Remove old contents if they exist
        if out_img_dir.exists():
            shutil.rmtree(out_img_dir)
        if out_lab_dir.exists():
            shutil.rmtree(out_lab_dir)

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

    # Copy files into splits
    for split, (start, end) in split_indices.items():
        out_img_dir = PROCESSED_DIR / split / "images"
        out_lab_dir = PROCESSED_DIR / split / "labels"

        for img_path, label_path in all_items[start:end]:
            stem = img_path.stem
            new_img_path = out_img_dir / img_path.name
            new_label_path = out_lab_dir / f"{stem}.txt"

            shutil.copy2(img_path, new_img_path)
            shutil.copy2(label_path, new_label_path)

    print("Finished creating train/val/test splits under data/processed.")


if __name__ == "__main__":
    main()
