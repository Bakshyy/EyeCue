import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

IMG_SIZE = 224
MODEL_PATH = "models/mobilenetv2_detector_best_fp32.tflite"
LABELS_DIR = Path("data/labels")
IMAGES_ROOT = Path("data/images")


def load_tflite(model_path):
    print(f"[INFO] Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[INFO] Input:  {input_details}")
    print(f"[INFO] Output: {output_details}")
    return interpreter, input_details, output_details


def run_inference(interpreter, input_details, output_details, img_bgr):
    # Same preprocessing as live_tflite_webcam.py
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    inp = resized.astype(np.float32)  # Do NOT /255; model has true_divide inside
    inp = np.expand_dims(inp, axis=0)

    input_index = input_details[0]["index"]
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()

    bbox_index = output_details[0]["index"]
    has_ball_index = output_details[1]["index"]

    pred_box = interpreter.get_tensor(bbox_index)[0]       # (4,)
    pred_has_ball = float(interpreter.get_tensor(has_ball_index)[0][0])

    return pred_box, pred_has_ball, resized  # resized RGB


def draw_box_cxcywh(img, box, color, label_text=None):
    """
    Interpret `box` as (cx, cy, w, h) normalized in [0,1].
    Draw on `img` (assumed square 224×224 or similar).
    """
    h, w, _ = img.shape
    cx, cy, bw, bh = box

    cx = np.clip(cx, 0.0, 1.0) * w
    cy = np.clip(cy, 0.0, 1.0) * h
    bw = np.clip(bw, 0.0, 1.0) * w
    bh = np.clip(bh, 0.0, 1.0) * h

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label_text is not None:
        cv2.putText(
            img,
            label_text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to visualize (train or val)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many random samples to show",
    )
    args = parser.parse_args()

    labels_csv = LABELS_DIR / args.split / "labels.csv"
    if not labels_csv.exists():
        print(f"[ERROR] Labels CSV not found: {labels_csv}")
        return

    df = pd.read_csv(labels_csv)
    if not {"filename", "cx", "cy", "w", "h", "has_ball"}.issubset(df.columns):
        print("[ERROR] Expected columns: filename, cx, cy, w, h, has_ball")
        print(f"Found columns: {list(df.columns)}")
        return

    img_dir = IMAGES_ROOT / args.split
    if not img_dir.exists():
        print(f"[ERROR] Images directory not found: {img_dir}")
        return

    interpreter, input_details, output_details = load_tflite(MODEL_PATH)

    indices = list(range(len(df)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    print(f"[INFO] Visualizing {len(indices)} samples from split '{args.split}'")

    for idx in indices:
        row = df.iloc[idx]
        fname = row["filename"]

        gt_box = np.array(
            [row["cx"], row["cy"], row["w"], row["h"]],
            dtype=np.float32,
        )
        gt_has_ball = float(row["has_ball"])

        img_path = img_dir / fname
        if not img_path.exists():
            print(f"[WARN] Missing image: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        # Run the same preprocessing + inference as webcam
        pred_box, pred_prob, resized_rgb = run_inference(
            interpreter, input_details, output_details, img_bgr
        )

        # Work on 224×224 BGR image for display
        vis = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

        # Draw GT (RED) and prediction (GREEN)
        draw_box_cxcywh(vis, gt_box, (0, 0, 255), label_text=f"GT:{gt_has_ball:.0f}")
        draw_box_cxcywh(vis, pred_box, (0, 255, 0), label_text=f"P:{pred_prob:.2f}")

        cv2.imshow(f"{args.split} sample (r=GT, g=PRED)", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
