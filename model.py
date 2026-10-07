"""Model architecture definitions for deepfake detection.

Provides CNN-LSTM hybrid models with configurable backbones:
- "light": Custom 4-block CNN feature extractor with momentum=0.9 BatchNorm.
- "mobilenetv2": MobileNetV2 feature extractor with optional ImageNet weights.
"""

from typing import Optional, Tuple
import numpy as np
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
    filters_list = [16, 32, 64, 128]
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


@tf.keras.utils.register_keras_serializable(package="DeepfakeDetector")
class TemporalAttention(layers.Layer):
    """Temporal attention pooling layer over recurrent sequence representations.

    Computes normalized attention coefficients alpha_t across frames:
        u_t = tanh(W * h_t + b)
        score_t = v^T * u_t
        alpha = softmax(score, axis=temporal_axis)
        context = sum_t (alpha_t * h_t)
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.w = self.add_weight(
            name="attn_w",
            shape=(dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="attn_v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, dim)
        u = tf.tanh(tf.tensordot(inputs, self.w, axes=1) + self.b)
        score = tf.tensordot(u, self.v, axes=1)  # (batch_size, seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)    # (batch_size, seq_len, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)  # (batch_size, dim)
        return context

    def compute_attention(self, inputs):
        """Compute normalized attention weights across frames: shape (batch_size, seq_len)."""
        u = tf.tanh(tf.tensordot(inputs, self.w, axes=1) + self.b)
        score = tf.tensordot(u, self.v, axes=1)
        weights = tf.nn.softmax(score, axis=1)
        return tf.squeeze(weights, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model(
    seq_len: int = 10,
    img_size: int = 128,
    backbone: str = "light",
    pretrained: bool = False,
    lstm_units: int = 128,
    dropout: float = 0.4,
    use_bilstm: bool = False,
    use_attention: bool = False,
    attention_units: int = 64,
) -> models.Model:
    """Build end-to-end CNN-LSTM / CNN-BiLSTM-Attention model for deepfake video detection.

    Pipeline:
        Input(uint8)
        -> Rescaling(1/127.5, offset=-1)
        -> TimeDistributed(CNN)
        -> Dropout
        -> LSTM or BiLSTM
        -> Optional TemporalAttention
        -> Dropout
        -> Dense(128, activation="relu")
        -> Dense(1, activation="sigmoid")

    Args:
        seq_len: Number of frames per video clip.
        img_size: Spatial resolution (H = W = img_size).
        backbone: Backbone type ("light" or "mobilenetv2").
        pretrained: Whether to use ImageNet pretrained weights (MobileNetV2).
        lstm_units: Hidden dimensionality of LSTM layer.
        dropout: Dropout rate applied before and after LSTM.
        use_bilstm: If True, uses Bidirectional(LSTM).
        use_attention: If True, applies TemporalAttention pooling.
        attention_units: Hidden dimensionality for attention scoring.

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

    if use_bilstm:
        x = layers.Bidirectional(
            layers.LSTM(units=lstm_units, return_sequences=use_attention),
            name="temporal_bilstm",
        )(x)
    else:
        x = layers.LSTM(units=lstm_units, return_sequences=use_attention, name="temporal_lstm")(x)

    if use_attention:
        x = TemporalAttention(units=attention_units, name="temporal_attention")(x)

    x = layers.Dropout(rate=dropout, name="dropout_2")(x)
    x = layers.Dense(units=128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(rate=0.3, name="dropout_3")(x)
    out = layers.Dense(units=1, activation="sigmoid", name="prediction")(x)

    tag = "v2" if (use_bilstm or use_attention) else backbone
    model = models.Model(inputs=seq_input, outputs=out, name=f"deepfake_detector_{tag}")
    return model


def build_model_v2(
    seq_len: int = 10,
    img_size: int = 128,
    backbone: str = "light",
    pretrained: bool = False,
    lstm_units: int = 128,
    dropout: float = 0.4,
    attention_units: int = 64,
) -> models.Model:
    """Build V2 model: CNN -> BiLSTM -> Temporal Attention -> Dense -> Sigmoid."""
    return build_model(
        seq_len=seq_len,
        img_size=img_size,
        backbone=backbone,
        pretrained=pretrained,
        lstm_units=lstm_units,
        dropout=dropout,
        use_bilstm=True,
        use_attention=True,
        attention_units=attention_units,
    )


def extract_temporal_attention(
    model: tf.keras.Model,
    clips: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract per-frame temporal attention weights from a model containing TemporalAttention.

    Args:
        model: Trained Keras model containing a TemporalAttention layer.
        clips: Array of video clips with shape (N, seq_len, H, W, 3) or (seq_len, H, W, 3).

    Returns:
        Array of attention weights of shape (N, seq_len) normalized to sum to 1.0 per clip,
        or None if model does not contain a TemporalAttention layer.
    """
    try:
        attn_layer = model.get_layer("temporal_attention")
    except ValueError:
        return None

    clips_arr = np.asarray(clips)
    is_single = False
    if clips_arr.ndim == 4:
        clips_arr = np.expand_dims(clips_arr, axis=0)
        is_single = True

    try:
        feat_model = tf.keras.Model(inputs=model.input, outputs=attn_layer.input)
        recurrent_feats = feat_model.predict(clips_arr, verbose=0)
        weights = attn_layer.compute_attention(recurrent_feats)
        weights_np = weights.numpy() if hasattr(weights, "numpy") else np.array(weights)
        if is_single:
            return weights_np[0]
        return weights_np
    except Exception as e:
        print(f"[WARNING] Attention extraction failed: {e}")
        return None


def transfer_cnn_weights(source_model: tf.keras.Model, target_model: tf.keras.Model) -> bool:
    """Transfer TimeDistributed CNN weights from a trained source model to a target model."""
    try:
        src_td = source_model.get_layer("time_distributed_cnn")
        tgt_td = target_model.get_layer("time_distributed_cnn")
        tgt_td.layer.set_weights(src_td.layer.get_weights())
        return True
    except Exception as e:
        print(f"[WARNING] CNN weight transfer failed: {e}")
        return False


if __name__ == "__main__":
    m = build_model()
    m.summary()
    mv2 = build_model_v2()
    mv2.summary()

