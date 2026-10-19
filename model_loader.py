"""
NammaRetina - Model Loader
Loads the trained EfficientNetB0 model for DR severity prediction.
Optionally downloads weights from MODEL_URL when the local file is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from config import MODEL_PATH, MODEL_URL, IMG_SIZE, NUM_CLASSES, DR_CLASSES

logger = logging.getLogger(__name__)

_model = None
_model_available = None


def _drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common share / uc URL forms."""
    if not url:
        return None
    parsed = urlparse(url)
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
    if path.exists() and path.stat().st_size > 0:
        return path

    url = MODEL_URL
    if not url:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Model missing at %s — downloading from MODEL_URL…", path)

    try:
        file_id = _drive_file_id(url)
        if file_id:
            import gdown

            gdown.download(id=file_id, output=str(path), quiet=False)
        else:
            import urllib.request

            urllib.request.urlretrieve(url, str(path))
    except Exception as exc:
        logger.error("Failed to download model from MODEL_URL: %s", exc)
        if path.exists() and path.stat().st_size == 0:
            try:
                path.unlink()
            except OSError:
                pass
        return path

    if path.exists() and path.stat().st_size > 0:
        logger.info("Model downloaded (%d bytes) → %s", path.stat().st_size, path)
    else:
        logger.error("Download finished but model file is missing/empty: %s", path)
    return path


def load_model():
    """Load the trained Keras model (singleton). Returns None if unavailable."""
    global _model, _model_available

    if _model_available is not None:
        return _model

    model_file = ensure_model_file(MODEL_PATH)
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

        _model = tf.keras.models.load_model(str(model_file))
        _model_available = True
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return _model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        _model_available = False
        _model = None
        return None


def is_model_available() -> bool:
    """Check whether the trained model has been loaded."""
    if _model_available is None:
        load_model()
    return bool(_model_available)


def preprocess_image(image_path: str) -> np.ndarray:
    """Read and preprocess an image for model inference.

    Returns a (1, IMG_SIZE, IMG_SIZE, 3) float32 array normalised to [-1, 1],
    matching the improved Colab training pipeline (img / 127.5 - 1.0).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
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

    logger.warning("No trained model available — returning neutral fallback (no prediction).")
    return {
        "severity": 0,
        "severity_label": DR_CLASSES.get(0, "Unknown"),
        "confidence": 0.0,
        "probabilities": [0.0] * NUM_CLASSES,
        "model_used": False,
    }
