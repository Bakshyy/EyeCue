import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# Image size for MobileNetV2
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# Allowed image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(images_dir: str) -> List[str]:
    """Return sorted list of image file paths in a directory."""
    if not os.path.isdir(images_dir):
        print(f"[WARN] Images dir does not exist: {images_dir}")
        return []

    files = []
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(images_dir, fname))

    files.sort()
    if not files:
        print(f"[WARN] No image files found in {images_dir}")
    return files


def _load_yolo_for_image(img_path: str, labels_dir: str) -> Tuple[List[float], float]:
    """
    Given a single image path and labels directory, load YOLO label if present.

    YOLO txt format (normalized):
        class_id x_center y_center width height

    Returns:
        bbox: [xmin, ymin, xmax, ymax] normalized to [0,1]
        has_ball: 1.0 if label present, else 0.0
    """
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base + ".txt")

    # Default: no ball
    if not os.path.isfile(label_path):
        return [0.0, 0.0, 0.0, 0.0], 0.0

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        # Empty file → treat as background
        return [0.0, 0.0, 0.0, 0.0], 0.0

    # Take the first object only (single ball case)
    parts = lines[0].split()
    if len(parts) < 5:
        print(f"[WARN] Malformed YOLO label (expected ≥5 values) in {label_path}")
        return [0.0, 0.0, 0.0, 0.0], 0.0

    # class_id = int(parts[0])  # we don't actually use the class id yet
    _, cx, cy, w, h = map(float, parts[:5])

    # Convert from YOLO (cx, cy, w, h) → (xmin, ymin, xmax, ymax)
    xmin = cx - w / 2.0
    xmax = cx + w / 2.0
    ymin = cy - h / 2.0
    ymax = cy + h / 2.0

    # Clamp just in case
    xmin = float(max(0.0, min(1.0, xmin)))
    xmax = float(max(0.0, min(1.0, xmax)))
    ymin = float(max(0.0, min(1.0, ymin)))
    ymax = float(max(0.0, min(1.0, ymax)))

    bbox = [xmin, ymin, xmax, ymax]
    has_ball = 1.0

    return bbox, has_ball


def load_annotations_from_yolo(
    images_dir: str, labels_dir: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Scan all images in images_dir, and for each image look for a YOLO txt
    in labels_dir. If not found, mark has_ball=0 and bbox=0s.
    """
    img_paths = _list_images(images_dir)
    if not img_paths:
        return [], np.zeros((0, 4), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    bboxes: List[List[float]] = []
    has_balls: List[List[float]] = []

    if not os.path.isdir(labels_dir):
        print(f"[WARN] Labels dir does not exist: {labels_dir}")
        print("[WARN] Treating all images as background (no ball).")
        for _ in img_paths:
            bboxes.append([0.0, 0.0, 0.0, 0.0])
            has_balls.append([0.0])
    else:
        for img_path in img_paths:
            bbox, has_ball = _load_yolo_for_image(img_path, labels_dir)
            bboxes.append(bbox)
            has_balls.append([has_ball])

    bboxes_arr = np.asarray(bboxes, dtype=np.float32)
    has_ball_arr = np.asarray(has_balls, dtype=np.float32)

    return img_paths, bboxes_arr, has_ball_arr


def load_example(img_path: tf.Tensor, labels: dict):
    """
    Given:
      img_path: scalar string tensor
      labels: dict with 'bbox' and 'has_ball'
    Returns:
      image tensor of shape (224, 224, 3), float32
      labels dict unchanged
    """
    # Ensure string dtype (defensive)
    img_path = tf.cast(img_path, tf.string)

    # Read file
    img_bytes = tf.io.read_file(img_path)

    # Decode image and force 3 channels
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    # Resize
    img = tf.image.resize(img, IMG_SIZE)

    # Cast to float32; normalization happens inside the model
    img = tf.cast(img, tf.float32)

    return img, labels


def make_dataset(split: str, shuffle: bool = True, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset for 'train' or 'val'.

    Expects structure:
      data/images/{split}/*.jpg|*.png|*.jpeg|*.bmp
      data/labels/{split}/*.txt   (YOLO format)
    """
    images_dir = os.path.join("data", "images", split)
    labels_dir = os.path.join("data", "labels", split)

    img_paths, bboxes, has_balls = load_annotations_from_yolo(images_dir, labels_dir)

    if len(img_paths) == 0:
        # Fail loudly instead of letting TF blow up with a weird type error
        raise RuntimeError(
            f"No samples found for split '{split}'. "
            f"Check that images are in {images_dir} and YOLO labels in {labels_dir}."
        )

    # NumPy 2.0: use Python str, not np.string_
    img_paths_arr = np.asarray(img_paths, dtype=str)

    labels_dict = {
        "bbox": bboxes,
        "has_ball": has_balls,
    }

    ds = tf.data.Dataset.from_tensor_slices((img_paths_arr, labels_dict))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths))

    ds = ds.map(
        lambda img_p, lab: load_example(img_p, lab),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds

