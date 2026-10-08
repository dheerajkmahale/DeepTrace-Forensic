"""Configuration module for deepfake detector pipeline.

Centralizes hyperparameters and path settings for preprocessing,
dataset generation, model architecture, training, and evaluation.
"""

from dataclasses import dataclass, asdict
import hashlib
import os
from typing import Optional, Tuple
import urllib.request

# Production Model Artifact Delivery Settings
PRODUCTION_MODEL_URL: str = (
    "https://github.com/dheerajkmahale/deepfake-detector/releases/download/v2.0.0-model/best_model.keras"
)
PRODUCTION_MODEL_SHA256: str = (
    "20b2d5dcf6300f4b001aebc7f73142e29ef6e06bbe55610062dea23425da8689"
)
PRODUCTION_MODEL_PATH: str = "outputs/best_model.keras"


def compute_sha256(filepath: str) -> str:
    """Compute the SHA256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def ensure_production_model(
    model_path: str = PRODUCTION_MODEL_PATH,
    expected_sha256: str = PRODUCTION_MODEL_SHA256,
    download_url: str = PRODUCTION_MODEL_URL,
) -> Tuple[bool, str]:
    """Ensure the verified production model checkpoint is available on disk.

    Verifies existing local model against expected SHA256. If missing or
    corrupted, downloads the production release artifact and validates checksum.

    Returns:
        (success: bool, status_message: str)
    """
    # 1. Check if model already exists and verify its integrity
    if os.path.exists(model_path):
        try:
            actual_hash = compute_sha256(model_path)
            if actual_hash.lower() == expected_sha256.lower():
                return True, "Production model verified locally."
            else:
                try:
                    os.remove(model_path)
                except Exception:
                    return (
                        False,
                        f"Existing model SHA256 mismatch ({actual_hash[:8]}... vs expected {expected_sha256[:8]}...) and unable to replace.",
                    )
        except Exception as e:
            return False, f"Error validating existing model file: {e}"

    # 2. Download from official verified GitHub Release asset
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp_path = f"{model_path}.tmp"

    try:
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "Mozilla/5.0 DeepTrace-Production-Model-Fetcher"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            with open(tmp_path, "wb") as out_f:
                while chunk := resp.read(65536):
                    out_f.write(chunk)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False, f"Failed to download production model from release: {e}"

    # 3. Verify SHA256 of downloaded file
    try:
        downloaded_hash = compute_sha256(tmp_path)
        if downloaded_hash.lower() != expected_sha256.lower():
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return (
                False,
                f"Checksum verification failed: expected {expected_sha256}, got {downloaded_hash}",
            )
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False, f"Error computing checksum of downloaded model: {e}"

    # 4. Atomic replacement
    try:
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception:
                pass
        os.replace(tmp_path, model_path)
    except Exception as e:
        return False, f"Error saving verified model artifact: {e}"

    return (
        True,
        f"Production model successfully downloaded and verified (SHA256: {downloaded_hash[:8]}...).",
    )


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
