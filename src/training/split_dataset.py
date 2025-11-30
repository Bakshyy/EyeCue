# src/training/split_dataset.py
# Split "all" images+labels into train/val/test folders.

import shutil
from pathlib import Path
import random

BASE_DIR = Path("data") / "processed"

ALL_IMAGES_DIR = BASE_DIR / "all" / "images"
ALL_LABELS_DIR = BASE_DIR / "all" / "labels"

SPLITS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}

# allowed image extensions
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]


def main():
    random.seed(42)

    # collect all image files that have label files
    all_items = []
    for img_path in sorted(ALL_IMAGES_DIR.rglob("*")):
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in EXTS:
            continue

        stem = img_path.stem
        label_path = ALL_LABELS_DIR / f"{stem}.txt"
        if not label_path.exists():
            # if label missing, skip
            # print("No label for", img_path)
            continue

        all_items.append((img_path, label_path))

    total = len(all_items)
    print(f"Total images: {total}")

    # shuffle
    random.shuffle(all_items)

    # compute split sizes
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

    # create output dirs
    for split in ["train", "val", "test"]:
        (BASE_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # copy files into splits
    for split, (start, end) in split_indices.items():
        out_img_dir = BASE_DIR / split / "images"
        out_lab_dir = BASE_DIR / split / "labels"

        for img_path, label_path in all_items[start:end]:
            stem = img_path.stem
            new_img_path = out_img_dir / img_path.name
            new_label_path = out_lab_dir / f"{stem}.txt"

            shutil.copy2(img_path, new_img_path)
            shutil.copy2(label_path, new_label_path)

    print("Finished creating train/val/test splits.")


if __name__ == "__main__":
    main()
