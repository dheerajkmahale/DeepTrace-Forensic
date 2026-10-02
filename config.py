"""Configuration module for deepfake detector pipeline.

Centralizes hyperparameters and path settings for preprocessing,
dataset generation, model architecture, training, and evaluation.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    """Default configuration for deepfake detector pipeline."""

    # Video & Preprocessing
    seq_len: int = 10
    img_size: int = 128
    clips_per_video: int = 3
    face_margin: float = 0.25
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    # Training Hyperparameters
    batch_size: int = 8
    epochs: int = 15
    lr: float = 1e-3
    lstm_units: int = 128
    dropout: float = 0.4
    backbone: str = "light"  # Options: "light", "mobilenetv2"
    pretrained: bool = False

    # Directories
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "outputs"

    def to_dict(self):
        """Convert configuration to dictionary."""
        return asdict(self)
