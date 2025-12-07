# src/debug/compare_keras_vs_tflite.py

from pathlib import Path
import numpy as np
import tensorflow as tf

from src.training.dataset_loader import make_dataset

MODELS_DIR = Path("models")
KERAS_MODEL_PATH = MODELS_DIR / "mobilenetv2_detector_best.keras"
TFLITE_FP32_PATH = MODELS_DIR / "mobilenetv2_detector_best_fp32.tflite"


def load_tflite_interpreter(model_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        print("[INFO] Using tflite_runtime.Interpreter")
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        print("[INFO] Using tf.lite.Interpreter")

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def get_output_indices(interp):
    out_details = interp.get_output_details()
    bbox_index = None
    has_ball_index = None
    for od in out_details:
        num_elems = int(np.prod(od["shape"]))
        if num_elems == 4:
            bbox_index = od["index"]
        elif num_elems == 1:
            has_ball_index = od["index"]
    if bbox_index is None or has_ball_index is None:
        raise RuntimeError(
            f"Could not find bbox/has_ball outputs. Shapes: "
            f"{[od['shape'] for od in out_details]}"
        )
    return bbox_index, has_ball_index


def main():
    # 1) Load Keras model
    print("[CHECK] Loading Keras model...")
    keras_model = tf.keras.models.load_model(
        str(KERAS_MODEL_PATH), safe_mode=False
    )

    # 2) Load TFLite model
    print("[CHECK] Loading TFLite model...")
    interp = load_tflite_interpreter(TFLITE_FP32_PATH)
    in_details = interp.get_input_details()
    bbox_idx, has_ball_idx = get_output_indices(interp)

    # 3) Build val dataset (no shuffle)
    print("[CHECK] Building val dataset...")
    ds_val = make_dataset("val", batch_size=1, shuffle=False)

    # 4) Compare a few samples
    for i, (images, targets) in enumerate(ds_val.take(10)):
        img = images.numpy()  # (1,224,224,3) float32 [0..255]
        true_bbox = targets["bbox"].numpy()[0]
        true_has = float(targets["has_ball"].numpy()[0, 0])

        # --- Keras ---
        keras_out = keras_model(img, training=False)
        # keras_out is a dict with keys "bbox" and "has_ball"
        pred_keras_bbox = keras_out["bbox"].numpy()[0]
        pred_keras_has = float(keras_out["has_ball"].numpy()[0, 0])

        # --- TFLite ---
        interp.set_tensor(in_details[0]["index"], img)
        interp.invoke()

        out_details = interp.get_output_details()

        bbox_raw = interp.get_tensor(bbox_idx)  # [1,4]
        has_raw = interp.get_tensor(has_ball_idx)  # [1,1]

        bbox_info = next(od for od in out_details if od["index"] == bbox_idx)
        has_info = next(od for od in out_details if od["index"] == has_ball_idx)

        # bbox
        if bbox_info["dtype"] == np.float32:
            pred_tflite_bbox = bbox_raw[0].astype(np.float32)
        else:
            scale, zero_point = bbox_info["quantization"]
            pred_tflite_bbox = (bbox_raw.astype(np.float32) - zero_point) * scale
            pred_tflite_bbox = pred_tflite_bbox[0]

        # has_ball
        if has_info["dtype"] == np.float32:
            pred_tflite_has = float(has_raw[0, 0])
        else:
            scale, zero_point = has_info["quantization"]
            pred_tflite_has = float(
                (has_raw.astype(np.float32) - zero_point) * scale
            )[0, 0]

        print(f"\n===== Sample {i} =====")
        print(f"GT has_ball: {true_has:.0f},  GT bbox: {true_bbox}")
        print(f"Keras has_ball:  {pred_keras_has:.3f}, bbox: {pred_keras_bbox}")
        print(f"TFLite has_ball: {pred_tflite_has:.3f}, bbox: {pred_tflite_bbox}")

    print("\n[CHECK] Done.")


if __name__ == "__main__":
    main()

