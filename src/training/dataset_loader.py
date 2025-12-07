# src/training/dataset_loader.py

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Final input size for MobileNetV2
IMG_SIZE = (224, 224)

# Base directory for processed data
PROCESSED_BASE = Path("data") / "processed"

# Allowed image extensions
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# ---------------------------------------------------------------------------
# Helpers to load labels.csv and image paths
# ---------------------------------------------------------------------------

def _load_split_labels(split: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load image paths, bounding boxes, and has_ball labels for a given split.

    Returns:
        img_paths: list of absolute path strings for each image
        bboxes:    np.ndarray of shape (N, 4)  with [cx, cy, w, h] in [0, 1]
        has_ball:  np.ndarray of shape (N,)    with 0 or 1
    """
    labels_dir = PROCESSED_BASE / split / "labels"
    images_dir = PROCESSED_BASE / split / "images"
    csv_path = labels_dir / "labels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found for split '{split}': {csv_path}"
        )

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images dir not found for split '{split}': {images_dir}"
        )

    img_paths: List[str] = []
    bboxes_list: List[List[float]] = []
    has_ball_list: List[int] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            img_path = images_dir / filename

            # Skip if image file does not exist (shouldn't happen after split_dataset)
            if not img_path.exists():
                # You can log this if you want to debug:
                # print(f"[WARN] Missing image file for row: {filename}")
                continue

            cx = float(row["cx"])
            cy = float(row["cy"])
            w = float(row["w"])
            h = float(row["h"])
            has_ball = int(row["has_ball"])

            img_paths.append(str(img_path))
            bboxes_list.append([cx, cy, w, h])
            has_ball_list.append(has_ball)

    if len(img_paths) == 0:
        raise RuntimeError(
            f"No valid samples found for split '{split}'. "
            f"Checked CSV at: {csv_path}"
        )

    bboxes = np.array(bboxes_list, dtype=np.float32)  # (N, 4)
    has_ball = np.array(has_ball_list, dtype=np.float32)  # (N,)

    print(
        f"[INFO] Loaded split '{split}': {len(img_paths)} samples "
        f"(has_ball=1: {(has_ball == 1).sum()}, has_ball=0: {(has_ball == 0).sum()})"
    )

    return img_paths, bboxes, has_ball


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def _build_dataset(
    img_paths: List[str],
    bboxes: np.ndarray,
    has_ball: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from lists/arrays.

    Output element spec:
      (image, targets)
        image:   float32 tensor [H, W, 3]
        targets: {'bbox': [4], 'has_ball': [1]}
    """

    num_samples = len(img_paths)

    # Base dataset from tensors
    ds = tf.data.Dataset.from_tensor_slices({
        "image_path": img_paths,
        "bbox": bboxes,
        "has_ball": has_ball,
    })

    def _load_and_preprocess(sample):
        # Read image file
        img_bytes = tf.io.read_file(sample["image_path"])

        # Decode without forcing channel count; preserve original
        img = tf.io.decode_image(
            img_bytes,
            channels=0,              # let TF use the file's real channels
            expand_animations=False,
        )
        # Ensure rank-3 (H, W, C)
        img.set_shape([None, None, None])

        # Normalize channels to 3 (RGB)
        def _fix_channels(x):
            c = tf.shape(x)[-1]

            # If grayscale (1 channel) -> RGB
            x = tf.cond(
                tf.equal(c, 1),
                lambda: tf.image.grayscale_to_rgb(x),
                lambda: x,
            )

            # If RGBA (4 channels) -> drop alpha
            def drop_alpha(y):
                return y[..., :3]

            x = tf.cond(
                tf.equal(tf.shape(x)[-1], 4),
                lambda: drop_alpha(x),
                lambda: x,
            )

            x.set_shape([None, None, 3])
            return x

        img = _fix_channels(img)

        # Cast to float32, still in [0, 255]
        img = tf.cast(img, tf.float32)

        # Resize to model input size
        img = tf.image.resize(img, IMG_SIZE)

        # Prepare targets
        bbox = tf.cast(sample["bbox"], tf.float32)         # shape (4,)
        has_ball_val = tf.cast(sample["has_ball"], tf.float32)  # scalar
        has_ball_val = tf.reshape(has_ball_val, [1])       # shape (1,)

        # Set static shapes for Keras
        img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
        bbox.set_shape((4,))
        has_ball_val.set_shape((1,))

        targets = {
            "bbox": bbox,
            "has_ball": has_ball_val,
        }
        return img, targets

    if shuffle:
        ds = ds.shuffle(
            buffer_size=num_samples,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        _load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_dataset(
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Public API used by train_mobilenetv2_detector.py

    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch size for training / eval
        shuffle: whether to shuffle samples

    Returns:
        tf.data.Dataset yielding (image, {'bbox': ..., 'has_ball': ...})
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

    img_paths, bboxes, has_ball = _load_split_labels(split)
    ds = _build_dataset(img_paths, bboxes, has_ball, batch_size=batch_size, shuffle=shuffle)

    # Debug: print element spec like you saw before
    print(f"[INFO] {split} element spec:", ds.element_spec)
    return ds
