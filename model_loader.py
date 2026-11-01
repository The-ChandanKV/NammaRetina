"""
NammaRetina - Model Loader
Loads the trained EfficientNetB0 model for DR severity prediction.
Optionally downloads weights from MODEL_URL when the local file is missing.
"""

from __future__ import annotations

import logging
import hashlib
import tempfile
import zipfile
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from config import MODEL_PATH, MODEL_URL, MODEL_SHA256, IMG_SIZE, NUM_CLASSES, DR_CLASSES

logger = logging.getLogger(__name__)

_model = None
_model_available = None
_model_sha256 = None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common share / uc URL forms."""
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.hostname not in {"drive.google.com", "docs.google.com"}:
        return None
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    # https://drive.google.com/file/d/<ID>/view
    parts = parsed.path.strip("/").split("/")
    if "d" in parts:
        i = parts.index("d")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def ensure_model_file(dest: str | Path | None = None) -> Path:
    """Return path to the model file, downloading from MODEL_URL if needed."""
    path = Path(dest or MODEL_PATH)
    if MODEL_SHA256 and (len(MODEL_SHA256) != 64 or any(c not in "0123456789abcdef" for c in MODEL_SHA256)):
        raise ValueError("MODEL_SHA256 must contain the model's 64-character SHA256 digest")
    if path.exists() and path.stat().st_size > 0 and (not MODEL_SHA256 or file_sha256(path) == MODEL_SHA256):
        return path

    url = MODEL_URL
    if not url:
        if path.exists() and MODEL_SHA256:
            raise ValueError("Local model checksum does not match MODEL_SHA256")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Model missing at %s — downloading from MODEL_URL…", path)

    handle, temporary_name = tempfile.mkstemp(prefix="model-download-", suffix=".keras", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        file_id = _drive_file_id(url)
        if file_id:
            import gdown

            gdown.download(id=file_id, output=str(temporary), quiet=False)
        else:
            import urllib.request

            import shutil
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("MODEL_URL did not return a valid .keras archive")
        if MODEL_SHA256 and file_sha256(temporary) != MODEL_SHA256:
            raise ValueError("Downloaded model checksum does not match MODEL_SHA256")
        temporary.replace(path)
    except Exception as exc:
        logger.error("Failed to download model from MODEL_URL: %s", exc)
        raise
    finally:
        temporary.unlink(missing_ok=True)

    if path.exists() and path.stat().st_size > 0:
        logger.info("Model downloaded (%d bytes) → %s", path.stat().st_size, path)
    else:
        logger.error("Download finished but model file is missing/empty: %s", path)
    return path


def load_model():
    """Load the trained Keras model (singleton). Returns None if unavailable."""
    global _model, _model_available, _model_sha256

    if _model_available is not None:
        return _model

    try:
        model_file = ensure_model_file(MODEL_PATH)
    except Exception as exc:
        logger.error("Model artifact unavailable: %s", exc)
        _model_available = False
        _model = None
        _model_sha256 = None
        return None
    if not model_file.exists() or model_file.stat().st_size == 0:
        logger.warning(
            "Trained model not found at %s. Using fallback predictions. "
            "Train with train_improved_dr_model.ipynb and set MODEL_URL.",
            MODEL_PATH,
        )
        _model_available = False
        _model = None
        return None

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(str(model_file), compile=False)
        if _model.input_shape != (None, IMG_SIZE, IMG_SIZE, 3) or _model.output_shape != (None, NUM_CLASSES):
            raise ValueError("Model input/output shapes are incompatible with this app")
        _model_sha256 = file_sha256(model_file)
        _model_available = True
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return _model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        _model_available = False
        _model = None
        _model_sha256 = None
        return None


def is_model_available() -> bool:
    """Check whether the trained model has been loaded."""
    if _model_available is None:
        load_model()
    return bool(_model_available)


def model_fingerprint():
    """Loaded artifact identity for production deployment verification."""
    return _model_sha256


def preprocess_image(image_path: str) -> np.ndarray:
    """Read and preprocess an image for model inference.

    Returns a (1, IMG_SIZE, IMG_SIZE, 3) float32 array normalised to [-1, 1],
    matching the improved Colab training pipeline (img / 127.5 - 1.0).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    from retinal_preprocessing import prepare_retina
    img = prepare_retina(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(img, axis=0)


def predict_severity(image_path: str) -> dict:
    """Run inference on a retinal image and return prediction details.

    Returns a dict with keys:
        severity (int): predicted DR class 0-4
        severity_label (str): human-readable label
        confidence (float): confidence score 0.0-1.0
        probabilities (list[float]): per-class probabilities
        model_used (bool): True if the real model was used
    """
    model = load_model()

    if model is not None:
        img = preprocess_image(image_path)
        predictions = model.predict(img, verbose=0)
        predictions = np.asarray(predictions)
        if predictions.shape != (1, NUM_CLASSES) or not np.all(np.isfinite(predictions)):
            raise ValueError("The model returned an invalid classification output.")
        if np.any(predictions < 0) or np.any(predictions > 1) or not np.isclose(predictions[0].sum(), 1, atol=0.01):
            raise ValueError("The model returned invalid class probabilities.")
        probabilities = predictions[0].tolist()
        severity = int(np.argmax(probabilities))
        confidence = float(probabilities[severity])
        return {
            "severity": severity,
            "severity_label": DR_CLASSES.get(severity, "Unknown"),
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": True,
        }

    logger.warning("No trained model available — analysis is unavailable.")
    return {
        "severity": None,
        "severity_label": "Analysis unavailable",
        "confidence": 0.0,
        "probabilities": [0.0] * NUM_CLASSES,
        "model_used": False,
    }
