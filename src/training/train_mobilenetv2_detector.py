import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .dataset_loader import make_dataset, IMG_SIZE, BATCH_SIZE


MODELS_DIR = os.path.join("models")
os.makedirs(MODELS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_detector_best.keras")


def build_mobilenetv2_detector(input_shape=(224, 224, 3)) -> keras.Model:
    """
    MobileNetV2 backbone with:
      - bbox regression head: 4 values (xmin, ymin, xmax, ymax)
      - has_ball classification head: 1 sigmoid
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Normalize to [0, 1]
    x = layers.Lambda(lambda t: t / 255.0, name="true_divide")(inputs)

    # MobileNetV2 backbone
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    base.trainable = False  # start frozen; you can unfreeze later

    x = base(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    # Heads
    bbox_out = layers.Dense(4, name="bbox")(x)  # regression head
    has_ball_out = layers.Dense(1, activation="sigmoid", name="has_ball")(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"bbox": bbox_out, "has_ball": has_ball_out},
        name="mobilenetv2_tennis_detector",
    )

    # Compile with separate losses
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "bbox": "mse",
            "has_ball": "binary_crossentropy",
        },
        loss_weights={
            "bbox": 1.0,
            "has_ball": 1.0,
        },
        metrics={
            "bbox": [keras.metrics.MeanAbsoluteError(name="mae")],
            "has_ball": [keras.metrics.BinaryAccuracy(name="acc")],
        },
    )

    model.summary()
    return model


def main():
    print("Building datasets...")
    train_ds = make_dataset("train", shuffle=True, batch_size=BATCH_SIZE)
    val_ds = make_dataset("val", shuffle=False, batch_size=BATCH_SIZE)

    print("Train element spec:", train_ds.element_spec)

    print("Creating model...")
    model = build_mobilenetv2_detector(input_shape=IMG_SIZE + (3,))

    # Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    EPOCHS = 40

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    # Save final model as well (even if not best)
    final_model_path = os.path.join(MODELS_DIR, "mobilenetv2_detector_final.keras")
    model.save(final_model_path)
    print(f"Training complete. Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
