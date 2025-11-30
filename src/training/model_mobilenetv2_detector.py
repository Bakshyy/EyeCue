# model_mobilenetv2_detector.py
# Defines a MobileNetV2-based model that outputs:
#  - has_ball (0..1)
#  - bbox (cx, cy, w, h)

import tensorflow as tf
from tensorflow.keras import layers, models

def create_mobilenetv2_detector(input_shape=(224, 224, 3)):
    # base model without top
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # for now, freeze base (can unfreeze later for fine-tuning)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # head 1: does the frame have a ball? (binary)
    has_ball_output = layers.Dense(1, activation="sigmoid", name="has_ball")(x)

    # head 2: bounding box (cx, cy, w, h)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox")(x)

    model = models.Model(
        inputs=base_model.input,
        outputs={
            "has_ball": has_ball_output,
            "bbox": bbox_output
        }
    )

    return model

def compile_detector(model, learning_rate=1e-4):
    losses = {
        "has_ball": tf.keras.losses.BinaryCrossentropy(),
        "bbox": tf.keras.losses.MeanSquaredError()
    }

    loss_weights = {
        "has_ball": 1.0,
        "bbox": 10.0  # give more weight to bbox error
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "has_ball": [tf.keras.metrics.BinaryAccuracy()],
            "bbox": [tf.keras.metrics.MeanAbsoluteError()]
        }
    )

    return model
