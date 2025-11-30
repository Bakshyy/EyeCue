import argparse
import cv2
import numpy as np
import tensorflow as tf

from src.training.dataset_loader import make_dataset, IMG_SIZE

MODEL_PATH = "models/mobilenetv2_detector_best_fp32.tflite"


def load_tflite(model_path):
    print(f"[INFO] Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[INFO] Input details: {input_details}")
    print(f"[INFO] Output details: {output_details}")
    return interpreter, input_details, output_details


def run_inference(interpreter, input_details, output_details, img_rgb_224):
    """
    img_rgb_224: numpy array (224, 224, 3), float32 in [0, 255]
    """
    inp = img_rgb_224.astype(np.float32)
    inp = np.expand_dims(inp, axis=0)  # (1, 224, 224, 3)

    input_index = input_details[0]["index"]
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()

    bbox_index = output_details[0]["index"]
    has_ball_index = output_details[1]["index"]

    pred_box = interpreter.get_tensor(bbox_index)[0]          # (4,)
    pred_has_ball = float(interpreter.get_tensor(has_ball_index)[0][0])

    return pred_box, pred_has_ball


def draw_box_cxcywh(img, box, color, label_text=None):
    """
    Interpret `box` as (cx, cy, w, h) normalized in [0,1] and draw on img.
    img is (H, W, 3) BGR.
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
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many samples to show",
    )
    args = parser.parse_args()

    # Build dataset exactly like training: same preprocessing & labels
    print(f"[INFO] Building dataset for split '{args.split}'...")
    ds = make_dataset(args.split, shuffle=True, batch_size=1)

    interpreter, input_details, output_details = load_tflite(MODEL_PATH)

    count = 0
    for batch in ds:
        if count >= args.num_samples:
            break

        img_batch, targets_batch = batch
        # img_batch: (1, 224, 224, 3)
        # targets_batch: dict with 'bbox' (1,4) and 'has_ball' (1,1)

        img_tensor = img_batch[0]          # (224, 224, 3)
        gt_box_tensor = targets_batch["bbox"][0]
        gt_has_ball_tensor = targets_batch["has_ball"][0]

        # Convert to numpy
        img_np = img_tensor.numpy()        # likely float32 [0,255]
        gt_box = gt_box_tensor.numpy().astype(np.float32)
        gt_has_ball = float(gt_has_ball_tensor.numpy())

        # Run TFLite on the same 224x224 RGB image
        pred_box, pred_prob = run_inference(
            interpreter,
            input_details,
            output_details,
            img_np,
        )

        # Prepare image for display: convert RGB -> BGR & clamp to uint8
        img_disp = np.clip(img_np, 0, 255).astype(np.uint8)
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

        # Draw GT (RED) and prediction (GREEN)
        draw_box_cxcywh(
            img_disp,
            gt_box,
            (0, 0, 255),
            label_text=f"GT:{gt_has_ball:.0f}",
        )
        draw_box_cxcywh(
            img_disp,
            pred_box,
            (0, 255, 0),
            label_text=f"P:{pred_prob:.2f}",
        )

        cv2.imshow(f"{args.split} sample (RED=GT, GREEN=PRED)", img_disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

        count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
