"""Model architecture definitions for deepfake detection.

Provides CNN-LSTM hybrid models with configurable backbones:
- "light": Custom 4-block CNN feature extractor with momentum=0.9 BatchNorm.
- "mobilenetv2": MobileNetV2 feature extractor with optional ImageNet weights.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_light_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    bn_momentum: float = 0.9,
) -> models.Model:
    """Build a lightweight 4-block CNN for frame feature extraction.

    Args:
        input_shape: Shape of single frame (H, W, C).
        bn_momentum: Batch normalization momentum. Defaults to 0.9 to
            prevent lagging moving statistics on small datasets.

    Returns:
        Keras Model extracting feature vectors from single frames.
    """
    inp = layers.Input(shape=input_shape, name="frame_input")
    x = inp

    filters_list = [32, 64, 128, 128]
    for i, filters in enumerate(filters_list, start=1):
        x = layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding="same",
            name=f"conv_{i}",
        )(x)
        # Explicit momentum=0.9 per requirement to avoid calibration issues
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"bn_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    return models.Model(inputs=inp, outputs=x, name="light_cnn")


def build_mobilenetv2_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    pretrained: bool = False,
) -> models.Model:
    """Build MobileNetV2 feature extractor.

    Args:
        input_shape: Shape of single frame (H, W, C).
        pretrained: If True, load ImageNet pretrained weights.

    Returns:
        Keras Model extracting feature vectors from single frames.
    """
    weights = "imagenet" if pretrained else None
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling="avg",
    )
    base.trainable = True
    return base


def build_model(
    seq_len: int = 10,
    img_size: int = 128,
    backbone: str = "light",
    pretrained: bool = False,
    lstm_units: int = 128,
    dropout: float = 0.4,
) -> models.Model:
    """Build end-to-end CNN-LSTM model for deepfake video detection.

    Pipeline:
        Input(uint8)
        -> Rescaling(1/127.5, offset=-1)
        -> TimeDistributed(CNN)
        -> Dropout
        -> LSTM
        -> Dropout
        -> Dense(64, activation="relu")
        -> Dense(1, activation="sigmoid")

    Args:
        seq_len: Number of frames per video clip.
        img_size: Spatial resolution (H = W = img_size).
        backbone: Backbone type ("light" or "mobilenetv2").
        pretrained: Whether to use ImageNet pretrained weights (MobileNetV2).
        lstm_units: Hidden dimensionality of LSTM layer.
        dropout: Dropout rate applied before and after LSTM.

    Returns:
        Compiled or uncompiled Keras Model expecting uint8 clips.
    """
    frame_shape = (img_size, img_size, 3)

    if backbone.lower() == "light":
        cnn = build_light_cnn(input_shape=frame_shape, bn_momentum=0.9)
    elif backbone.lower() == "mobilenetv2":
        cnn = build_mobilenetv2_cnn(input_shape=frame_shape, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: '{backbone}'. Choose 'light' or 'mobilenetv2'.")

    # Sequence input expects uint8 tensor with shape (batch, seq_len, H, W, 3)
    seq_input = layers.Input(
        shape=(seq_len, img_size, img_size, 3),
        dtype="uint8",
        name="video_clip_input",
    )

    # Normalize to [-1.0, 1.0]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling")(seq_input)
    x = layers.TimeDistributed(cnn, name="time_distributed_cnn")(x)
    x = layers.Dropout(rate=dropout, name="dropout_1")(x)
    x = layers.LSTM(units=lstm_units, return_sequences=False, name="temporal_lstm")(x)
    x = layers.Dropout(rate=dropout, name="dropout_2")(x)
    x = layers.Dense(units=64, activation="relu", name="dense_64")(x)
    out = layers.Dense(units=1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=seq_input, outputs=out, name=f"deepfake_detector_{backbone}")
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
