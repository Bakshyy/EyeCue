"""
Compare Keras vs TFLite predictions on the validation set.

This tells us whether the TFLite model (and our run_inference logic)
matches the Keras model that trains well on val.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.training.dataset_loader import make_dataset, IMG_SIZE

# IMPORTANT: we created a Lambda layer (x / 255.0), which Keras treats as unsafe by default.
# We trust this model (it's our own code), so enable unsafe deserialization *for this process*.
tf.keras.config.enable_unsafe_deserialization()


def load_keras_model(path: Path) -> tf.keras.Model:
    print(f"[INFO] Loading Keras model from {path}")
    # safe_mode=False is required because of the Lambda layer.
    model = tf.keras.models.load_model(path, safe_mode=False)
    model.summary()
    return model


def load_tflite_interpreter(path: Path):
    print(f"[INFO] Loading TFLite model from {path}")
    # Try tflite-runtime first (for Pi style); then fallback to TF Lite
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[INFO] TFLite input details: {input_details}")
    print(f"[INFO] TFLite output details: {output_details}")

    return interpreter, input_details, output_details


def tflite_predict(interpreter, input_details, output_details, batch_np: np.ndarray):
    """
    batch_np: (N, 224, 224, 3) float32 in [0, 255]
    Returns:
      bbox_pred: (N, 4)
      has_ball_pred: (N,)
    """
    bboxes = []
    probs = []

    for i in range(batch_np.shape[0]):
        x = batch_np[i : i + 1]  # (1, 224, 224, 3)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

        bbox_pred = interpreter.get_tensor(output_details[0]["index"])[0]  # (4,)
        has_ball_pred = interpreter.get_tensor(output_details[1]["index"])[0, 0]  # scalar

        bboxes.append(bbox_pred)
        probs.append(has_ball_pred)

    return np.stack(bboxes, axis=0), np.array(probs)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Keras vs TFLite predictions on val split."
    )
    parser.add_argument(
        "--keras",
        type=str,
        default="models/mobilenetv2_detector_best.keras",
        help="Path to Keras model file.",
    )
    parser.add_argument(
        "--tflite",
        type=str,
        default="models/mobilenetv2_detector_best_fp32.tflite",
        help="Path to TFLite model file.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="How many batches of val data to compare.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for val dataset when comparing.",
    )
    args = parser.parse_args()

    keras_path = Path(args.keras)
    tflite_path = Path(args.tflite)

    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    # 1) Load Keras model and TFLite interpreter
    keras_model = load_keras_model(keras_path)
    interpreter, input_details, output_details = load_tflite_interpreter(tflite_path)

    # 2) Build val dataset exactly like training
    print("[INFO] Building val dataset...")
    val_ds = make_dataset("val", shuffle=False, batch_size=args.batch_size)

    # 3) Compare a few batches
    bbox_diffs = []
    prob_diffs = []

    num_batches_done = 0

    for batch in val_ds:
        images, labels = batch
        # images: (B, 224, 224, 3), float32 in [0, 255] (Lambda /255 is inside the model)
        # labels["bbox"]: (B, 4)
        # labels["has_ball"]: (B, 1)

        # Keras predictions
        keras_pred = keras_model(images, training=False)
        keras_bbox = keras_pred["bbox"].numpy()
        keras_prob = keras_pred["has_ball"].numpy().reshape(-1)

        # TFLite predictions (feed same images)
        images_np = images.numpy()
        tflite_bbox, tflite_prob = tflite_predict(
            interpreter, input_details, output_details, images_np
        )

        diff_bbox = np.mean(np.abs(keras_bbox - tflite_bbox))
        diff_prob = np.mean(np.abs(keras_prob - tflite_prob))

        bbox_diffs.append(diff_bbox)
        prob_diffs.append(diff_prob)

        print(
            f"[BATCH {num_batches_done}] "
            f"mean |bbox_keras - bbox_tflite| = {diff_bbox:.4f}, "
            f"mean |prob_keras - prob_tflite| = {diff_prob:.4f}"
        )

        num_batches_done += 1
        if num_batches_done >= args.num_batches:
            break

    if bbox_diffs:
        print("\n[SUMMARY]")
        print(
            f"Avg abs bbox diff over {num_batches_done} batches: {np.mean(bbox_diffs):.44f}"
        )
        print(
            f"Avg abs prob diff over {num_batches_done} batches: {np.mean(prob_diffs):.4f}"
        )
    else:
        print("[WARN] No batches in val dataset? Check your data/labels.")


if __name__ == "__main__":
    main()
