"""
NammaRetina - Model Loader
Loads the trained EfficientNetB0 model for DR severity prediction.
Provides a graceful fallback when the model file is not available.
"""

from __future__ import annotations

import os
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from config import MODEL_PATH, IMG_SIZE, NUM_CLASSES, DR_CLASSES

logger = logging.getLogger(__name__)

_model = None
_model_available = None


def load_model():
    """Load the trained Keras model (singleton). Returns None if unavailable."""
    global _model, _model_available

    if _model_available is not None:
        return _model

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        logger.warning(
            "Trained model not found at %s. Using fallback predictions.", MODEL_PATH
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

    Returns a (1, IMG_SIZE, IMG_SIZE, 3) float32 array normalised to [0, 1].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
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

    # --- Fallback: deterministic pseudo-random based on image content ---
    logger.warning("Using fallback prediction (no trained model).")
    try:
        img = cv2.imread(image_path)
        if img is not None:
            # Use mean pixel intensity to seed a deterministic result
            seed_value = int(np.mean(img) * 100) % (NUM_CLASSES * 1000)
            rng = random.Random(seed_value)
            severity = rng.randint(0, NUM_CLASSES - 1)
        else:
            severity = 0
    except Exception:
        severity = 0

    return {
        "severity": severity,
        "severity_label": DR_CLASSES.get(severity, "Unknown"),
        "confidence": 0.0,
        "probabilities": [0.0] * NUM_CLASSES,
        "model_used": False,
    }
