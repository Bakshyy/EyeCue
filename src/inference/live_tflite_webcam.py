import argparse
import time

import cv2
import numpy as np
import tensorflow as tf

from src.training.dataset_loader import IMG_SIZE  # (224, 224)


DEFAULT_MODEL_PATH = "models/mobilenetv2_detector_best_fp32.tflite"


def load_tflite(model_path: str):
    print(f"[INFO] Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[INFO] Input details: {input_details}")
    print(f"[INFO] Output details: {output_details}")
    return interpreter, input_details, output_details


def run_inference(interpreter, input_details, output_details, img_bgr_224: np.ndarray):
    """
    img_bgr_224: (224, 224, 3) BGR uint8
    Our TFLite model expects float32 RGB in [0, 255],
    and does its own normalization internally (same as Keras model).
    """
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr_224, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Add batch dimension: (1, 224, 224, 3)
    inp = np.expand_dims(img_rgb, axis=0)

    input_index = input_details[0]["index"]
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()

    bbox_index = output_details[0]["index"]
    has_ball_index = output_details[1]["index"]

    pred_box = interpreter.get_tensor(bbox_index)[0]          # (4,)
    pred_has_ball = float(interpreter.get_tensor(has_ball_index)[0][0])

    return pred_box, pred_has_ball


def draw_box_cxcywh(img: np.ndarray, box, color, label_text=None):
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
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to TFLite model",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="has_ball probability threshold for drawing box",
    )
    args = parser.parse_args()

    interpreter, input_details, output_details = load_tflite(args.model)

    print(f"[INFO] Opening webcam index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam index {args.camera}")
        return

    target_w, target_h = IMG_SIZE[1], IMG_SIZE[0]

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera, exiting...")
            break

        # Optional: flip so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # Resize the BGR frame to 224x224 – exactly like training
        frame_resized = cv2.resize(frame, (target_w, target_h))

        # Run inference on the resized frame
        pred_box, pred_prob = run_inference(
            interpreter, input_details, output_details, frame_resized
        )

        disp = frame_resized.copy()

        if pred_prob >= args.prob_threshold:
            draw_box_cxcywh(
                disp,
                pred_box,
                (0, 255, 0),
                label_text=f"ball: {pred_prob:.2f}",
            )
        else:
            cv2.putText(
                disp,
                f"no ball ({pred_prob:.2f})",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # FPS calculation
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            disp,
            f"FPS: {fps:.1f}",
            (5, target_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("EyeCue TFLite Live (224x224)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

