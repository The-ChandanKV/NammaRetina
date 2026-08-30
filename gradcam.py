"""
NammaRetina - Grad-CAM Heatmap Generator
Generates Grad-CAM visualisations for the EfficientNetB0 DR model.
Falls back to a placeholder heatmap when the model is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import GRADCAM_LAYER_NAME, GRADCAM_COLORMAP, IMG_SIZE

logger = logging.getLogger(__name__)


def _apply_colormap(heatmap_gray: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Convert a single-channel float32 heatmap [0,1] to a BGR colour-mapped image."""
    cmap_lookup = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    cmap_id = cmap_lookup.get(colormap.lower(), cv2.COLORMAP_JET)
    heatmap_uint8 = np.uint8(255 * heatmap_gray)
    return cv2.applyColorMap(heatmap_uint8, cmap_id)


def generate_gradcam(
    model,
    image_path: str,
    predicted_class: int,
    save_dir: str,
    layer_name: str | None = None,
) -> str:
    """Generate and save a Grad-CAM heatmap overlay.

    Parameters
    ----------
    model : keras.Model or None
        The loaded Keras model. If None, a placeholder heatmap is generated.
    image_path : str
        Path to the original retinal image.
    predicted_class : int
        The predicted DR severity class index (0-4).
    save_dir : str
        Directory to save the heatmap image.
    layer_name : str, optional
        Target convolutional layer name for Grad-CAM.
        Defaults to ``config.GRADCAM_LAYER_NAME``.

    Returns
    -------
    str
        Absolute path to the saved heatmap PNG.
    """
    if layer_name is None:
        layer_name = GRADCAM_LAYER_NAME

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    img_stem = Path(image_path).stem
    heatmap_filename = f"{img_stem}_gradcam.png"
    output_path = save_path / heatmap_filename

    # Read the original image for overlay
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    if model is not None:
        try:
            heatmap = _compute_gradcam(model, image_path, predicted_class, layer_name)
        except Exception as exc:
            logger.warning("Grad-CAM computation failed (%s). Using placeholder.", exc)
            heatmap = _placeholder_heatmap(original_resized)
    else:
        logger.info("No model available — generating placeholder heatmap.")
        heatmap = _placeholder_heatmap(original_resized)

    # Resize heatmap to match the original image
    heatmap = cv2.resize(heatmap, (original_resized.shape[1], original_resized.shape[0]))

    # Apply colour map and blend with original
    coloured_heatmap = _apply_colormap(heatmap, GRADCAM_COLORMAP)
    overlay = cv2.addWeighted(original_resized, 0.6, coloured_heatmap, 0.4, 0)

    cv2.imwrite(str(output_path), overlay)
    logger.info("Heatmap saved to %s", output_path)
    return str(output_path)


def _find_conv_layer(model, layer_name: str):
    """Locate a layer by name, searching nested sub-models if needed.

    A model built with ``EfficientNetB0(input_tensor=...)`` keeps ``top_conv`` at
    the top level, so the direct lookup succeeds. But if the backbone is instead
    wrapped as a single nested sub-model layer, ``model.get_layer`` misses it;
    this falls back to scanning one level of nested sub-models. Returns the layer
    object, or ``None`` if it cannot be found anywhere.
    """
    try:
        return model.get_layer(layer_name)
    except Exception:
        pass
    for layer in getattr(model, "layers", []):
        if getattr(layer, "name", None) == layer_name:
            return layer
        if hasattr(layer, "layers"):  # nested sub-model (e.g. the EfficientNet backbone)
            try:
                return layer.get_layer(layer_name)
            except Exception:
                continue
    return None


def _compute_gradcam(
    model, image_path: str, predicted_class: int, layer_name: str
) -> np.ndarray:
    """Compute Grad-CAM heatmap using TensorFlow GradientTape.

    Returns a float32 array of shape (H, W) with values in [0, 1].
    """
    import tensorflow as tf

    # Preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img.astype(np.float32) / 127.5 - 1.0, axis=0)

    # Locate the target conv layer, searching nested sub-models as a fallback.
    conv_layer = _find_conv_layer(model, layer_name)
    if conv_layer is None:
        raise ValueError(
            f"Grad-CAM layer '{layer_name}' not found in model (including nested sub-models)."
        )

    # Build a sub-model that outputs the conv layer activations AND the predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Weighted sum of feature maps
    cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1)
    cam = tf.nn.relu(cam)

    # Normalise to [0, 1]
    heatmap = cam.numpy()[0]
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def _placeholder_heatmap(image: np.ndarray) -> np.ndarray:
    """Generate a simple placeholder heatmap based on image intensity.

    Creates a centre-weighted Gaussian-like heatmap as a reasonable
    visual placeholder when the real model is unavailable.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Create a radial gradient centred on the image
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    radial = 1.0 - np.clip(r / max_r, 0, 1)

    # Blend with actual image intensity for some spatial variation
    intensity = gray.astype(np.float32) / 255.0
    heatmap = 0.6 * radial + 0.4 * intensity
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.astype(np.float32)
