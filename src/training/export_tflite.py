from __future__ import annotations

from pathlib import Path
import tensorflow as tf

MODELS_DIR = Path("models")
KERAS_MODEL_PATH = MODELS_DIR / "mobilenetv2_detector_best.keras"

TFLITE_FP32_PATH = MODELS_DIR / "mobilenetv2_detector_best_fp32.tflite"
TFLITE_DR_PATH = MODELS_DIR / "mobilenetv2_detector_best_dynamic.tflite"  # dynamic range quant


def export_fp32(model: tf.keras.Model) -> None:
    """
    Plain float32 TFLite export.

    - Same math as the Keras model
    - No quantization, no extra optimizations
    """
    print("[EXPORT] Converting to TFLite (float32, no quant)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Make everything explicit
    converter.optimizations = []  # no graph-level optimizations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Explicit types
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    TFLITE_FP32_PATH.write_bytes(tflite_model)
    print(f"[EXPORT] Saved float32 TFLite model to: {TFLITE_FP32_PATH}")


def export_dynamic_range(model: tf.keras.Model) -> None:
    """
    Dynamic range quantization:
      - Weights quantized, activations stay float32
      - Same input/output dtypes as fp32 model
      - No representative dataset needed
    """
    print("[EXPORT] Converting to TFLite (dynamic range quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Keep float32 I/O so your inference code stays identical
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    TFLITE_DR_PATH.write_bytes(tflite_model)
    print(f"[EXPORT] Saved dynamic-range TFLite model to: {TFLITE_DR_PATH}")


def main() -> None:
    if not KERAS_MODEL_PATH.exists():
        print(f"[EXPORT] Error: Keras model not found at {KERAS_MODEL_PATH}")
        return

    print("[EXPORT] Loading Keras model...")
    # Lambda layer -> need unsafe deserialization (we trust our own model)
    model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), safe_mode=False)

    export_fp32(model)
    export_dynamic_range(model)

    print("[EXPORT] Done.")


if __name__ == "__main__":
    main()
