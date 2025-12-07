#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import sys

import numpy as np

# Make sure we can see system-wide packages (picamera2, etc.)
sys.path.append("/usr/lib/python3/dist-packages")

# Try to use tflite-runtime first (Pi), fall back to TensorFlow Lite if available.
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
    print("[INFO] Using tflite_runtime.Interpreter")
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    print("[INFO] Using tf.lite.Interpreter")

from picamera2 import Picamera2
import cv2

# ---------------------------------------------------------------------------
# Model / image settings
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("models/mobilenetv2_detector_best_fp32.tflite")


# ---------------------------------------------------------------------------
# Interpreter helpers
# ---------------------------------------------------------------------------

def load_interpreter(model_path: Path) -> Interpreter:
    print(f"[INFO] Loading TFLite model: {model_path}")
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[INFO] Input details: {input_details}")
    print(f"[INFO] Output details: {output_details}")

    return interpreter


def get_output_indices(interpreter: Interpreter):
    """
    Infer which output is bbox vs has_ball based on tensor shape:

      - bbox: 4 elements total   (e.g., [1, 4])
      - has_ball: 1 element      (e.g., [1, 1])
    """
    output_details = interpreter.get_output_details()
    bbox_index = None
    has_ball_index = None

    for od in output_details:
        num_elems = int(np.prod(od["shape"]))
        if num_elems == 4:
            bbox_index = od["index"]
        elif num_elems == 1:
            has_ball_index = od["index"]

    if bbox_index is None or has_ball_index is None:
        shapes = [od["shape"] for od in output_details]
        raise RuntimeError(f"Could not determine bbox/has_ball outputs from shapes: {shapes}")

    print(f"[INFO] bbox output index: {bbox_index}")
    print(f"[INFO] has_ball output index: {has_ball_index}")
    return bbox_index, has_ball_index


# ---------------------------------------------------------------------------
# Preprocessing (must match training)
# ---------------------------------------------------------------------------

def preprocess_frame(frame_rgb: np.ndarray, input_details):
    """
    frame_rgb: HxWx3, RGB uint8 [0..255] from PiCamera2.

    Training pipeline:
      - tf.io.decode_image -> RGB uint8 [0..255]
      - tf.cast to float32
      - model has Lambda(x/255.0) inside

    So here we:
      - resize to 224x224
      - keep RGB
      - cast to float32 [0..255] (no /255!)
      - handle quantization if the input is int8/uint8
    """
    # Resize to model input size
    resized = cv2.resize(frame_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    input_info = input_details[0]
    dtype = input_info["dtype"]
    scale, zero_point = input_info["quantization"]

    if dtype == np.float32:
        tensor = resized.astype(np.float32)  # [0..255], Lambda in model does /255
    else:
        if scale == 0:
            raise RuntimeError("Invalid quantization scale 0 for input tensor")
        # Map [0,255] float to quantized domain
        tensor = resized.astype(np.float32) / scale + zero_point

        if dtype == np.int8:
            tensor = np.clip(tensor, -128, 127).astype(np.int8)
        elif dtype == np.uint8:
            tensor = np.clip(tensor, 0, 255).astype(np.uint8)
        else:
            raise RuntimeError(f"Unsupported input dtype: {dtype}")

    tensor = np.expand_dims(tensor, axis=0)  # (1, 224, 224, 3)
    return tensor


# ---------------------------------------------------------------------------
# BBox decoding: [cx, cy, w, h] in [0,1] -> pixel box
# ---------------------------------------------------------------------------

def decode_bbox_normalized(bbox, frame_w: int, frame_h: int):
    """
    bbox: array-like [cx, cy, w, h] in [0,1]
    Returns (xmin, ymin, xmax, ymax) in pixel coordinates, clamped to frame size.
    """
    cx, cy, bw, bh = [float(v) for v in bbox]

    # Clamp to [0,1] to avoid weird outliers
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    bw = np.clip(bw, 0.0, 1.0)
    bh = np.clip(bh, 0.0, 1.0)

    cx_px = cx * frame_w
    cy_px = cy * frame_h
    bw_px = bw * frame_w
    bh_px = bh * frame_h

    xmin = int(cx_px - bw_px / 2.0)
    ymin = int(cy_px - bh_px / 2.0)
    xmax = int(cx_px + bw_px / 2.0)
    ymax = int(cy_px + bh_px / 2.0)

    xmin = max(0, min(frame_w - 1, xmin))
    ymin = max(0, min(frame_h - 1, ymin))
    xmax = max(0, min(frame_w - 1, xmax))
    ymax = max(0, min(frame_h - 1, ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    interpreter: Interpreter,
    input_details,
    bbox_index: int,
    has_ball_index: int,
    frame_rgb: np.ndarray,
    frame_w: int,
    frame_h: int,
):
    """
    Run one forward pass and return:

      bbox_px: (xmin, ymin, xmax, ymax) in pixel coords (or None)
      prob: float (0..1)
      infer_ms: inference time in ms
    """
    # Preprocess
    input_tensor = preprocess_frame(frame_rgb, input_details)

    # Set input
    interpreter.set_tensor(input_details[0]["index"], input_tensor)

    # Inference
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    infer_ms = (t1 - t0) * 1000.0

    # Get raw outputs
    out_details = interpreter.get_output_details()

    # bbox
    bbox_raw = interpreter.get_tensor(bbox_index)  # shape [1,4] (maybe quantized)
    bbox_info = next(od for od in out_details if od["index"] == bbox_index)
    if bbox_info["dtype"] == np.float32:
        bbox = bbox_raw[0].astype(np.float32)
    else:
        scale, zero_point = bbox_info["quantization"]
        bbox = (bbox_raw.astype(np.float32) - zero_point) * scale
        bbox = bbox[0]

    # has_ball
    has_ball_raw = interpreter.get_tensor(has_ball_index)  # shape [1,1]
    has_ball_info = next(od for od in out_details if od["index"] == has_ball_index)
    if has_ball_info["dtype"] == np.float32:
        prob = float(has_ball_raw[0, 0])
    else:
        scale, zero_point = has_ball_info["quantization"]
        prob = float((has_ball_raw.astype(np.float32) - zero_point) * scale[0, 0])

    # Decode bbox in pixel coords
    bbox_px = decode_bbox_normalized(bbox, frame_w, frame_h)

    return bbox_px, prob, infer_ms


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live TFLite inference using Raspberry Pi camera."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to TFLite model (.tflite).",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.5,
        help="Minimum probability to draw box.",
    )
    parser.add_argument(
        "--rotate180",
        action="store_true",
        help="Rotate camera frames 180 degrees (if camera is mounted upside-down).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror frames horizontally (to match training orientation).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="If set, do not open an OpenCV window (headless mode).",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    interpreter = load_interpreter(model_path)
    input_details = interpreter.get_input_details()
    bbox_index, has_ball_index = get_output_indices(interpreter)

    # Initialize PiCamera2
    print("[INFO] Initializing PiCamera2...")
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1.0)  # small warm-up

    print("[INFO] Starting live inference. Press 'q' in the OpenCV window to quit.")

    fps = 0.0
    frame_counter = 0
    fps_time_ref = time.time()

    try:
        while True:
            frame_rgb = picam.capture_array()  # HxWx3 RGB uint8

            if args.rotate180:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
            if args.mirror:
                frame_rgb = cv2.flip(frame_rgb, 1)

            frame_h, frame_w = frame_rgb.shape[:2]

            # Run model
            bbox_px, prob, infer_ms = run_inference(
                interpreter,
                input_details,
                bbox_index,
                has_ball_index,
                frame_rgb,
                frame_w,
                frame_h,
            )

            # Convert to BGR for display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # FPS calculation
            frame_counter += 1
            now = time.time()
            if now - fps_time_ref >= 1.0:
                fps = frame_counter / (now - fps_time_ref)
                frame_counter = 0
                fps_time_ref = now

            # Draw info text
            label_text = f"prob={prob:.2f}, inf={infer_ms:.1f} ms, FPS={fps:.1f}"
            cv2.putText(
                frame_bgr,
                label_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw bbox if confident
            if prob >= args.min_prob and bbox_px is not None:
                xmin, ymin, xmax, ymax = bbox_px
                cv2.rectangle(
                    frame_bgr,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 0),
                    2,
                )
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2
                cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

            if not args.no_display:
                cv2.imshow("EyeCue PiCam", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] 'q' pressed, exiting.")
                    break
            else:
                # Headless mode: log basic status
                print(f"prob={prob:.3f}, inf={infer_ms:.1f}ms, FPS={fps:.1f}", end="\r")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, exiting.")
    finally:
        picam.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()