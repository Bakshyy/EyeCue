# src/training/model_mobilenetv2_detector.py

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

from .dataset_loader import make_dataset

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224, 3)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEST_KERAS_PATH = MODELS_DIR / "mobilenetv2_detector_best.keras"
LAST_KERAS_PATH = MODELS_DIR / "mobilenetv2_detector_last.keras"

BATCH_SIZE = 32
EPOCHS = 40


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_mobilenetv2_detector(input_shape=IMG_SIZE) -> tf.keras.Model:
    """
    MobileNetV2 backbone with:
      - bbox head  -> [cx, cy, w, h] in [0, 1]  (sigmoid)
      - has_ball   -> probability in [0, 1]     (sigmoid)
    """

    # Backbone
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )

    # Start with backbone frozen; we can fine-tune later if needed
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # IMPORTANT: normalize here (this replaces the old Lambda in your previous model).
    # Dataset loader already gives float32 [0, 255]; we divide by 255 once.
    x = layers.Lambda(lambda z: z / 255.0, name="true_divide")(inputs)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    # BBOX head: [cx, cy, w, h] normalized to [0,1]
    bbox_out = layers.Dense(4, activation="sigmoid", name="bbox")(x)

    # has_ball head: probability
    has_ball_out = layers.Dense(1, activation="sigmoid", name="has_ball")(x)

    model = models.Model(
        inputs=inputs,
        outputs={"bbox": bbox_out, "has_ball": has_ball_out},
        name="mobilenetv2_tennis_detector",
    )

    # ---- Losses & metrics ----
    losses = {
        "bbox": tf.keras.losses.Huber(name="bbox_loss"),           # robust regression
        "has_ball": tf.keras.losses.BinaryCrossentropy(name="has_ball_loss"),
    }
    # Make bbox matter more than has_ball
    loss_weights = {
        "bbox": 5.0,
        "has_ball": 1.0,
    }
    metrics = {
        "bbox": [tf.keras.metrics.MeanAbsoluteError(name="bbox_mae")],
        "has_ball": [tf.keras.metrics.BinaryAccuracy(name="has_ball_accuracy")],
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("[TRAIN] Building datasets...")
    train_ds = make_dataset("train", batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_dataset("val", batch_size=BATCH_SIZE, shuffle=False)

    print("[TRAIN] Building model...")
    model = build_mobilenetv2_detector()
    model.summary()

    # Callbacks: checkpoint, early stopping, LR scheduler
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_KERAS_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print("[TRAIN] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    )

    print("[TRAIN] Saving last-epoch model...")
    model.save(str(LAST_KERAS_PATH))
    print(f"[TRAIN] Best model saved to: {BEST_KERAS_PATH}")
    print(f"[TRAIN] Last model saved to: {LAST_KERAS_PATH}")


if __name__ == "__main__":
    main()
