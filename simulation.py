"""Illustrative spatial attention visualization for retinal screens.

This module does NOT forecast where diabetic retinopathy will spread.
It produces a deterministic overlay of model-derived Grad-CAM attention,
optionally dilated or contracted from the patient's recorded ordinal trend,
with an anatomical field-of-view mask. Outputs must be labeled as illustrative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config

ILLUSTRATIVE_CAPTION = "Illustrative Spatial Simulation — Not a Clinical Forecast"
METHOD_DESCRIPTION = (
    "Deterministic Grad-CAM attention remapping with FOV masking and trend-based "
    "morphological dilation/erosion. No synthetic lesions are invented. "
    "Grad-CAM intensity does not predict future disease location."
)


def _status_spread_factor(status: str) -> float:
    """Map recorded ordinal trend labels to a fixed morphological scale."""
    if status == "Worsened":
        return 1.35
    if status == "Stable":
        return 1.08
    if status == "Improved":
        return 0.85
    return 1.05  # Initial / unknown


def _attention_rgb(values: np.ndarray) -> np.ndarray:
    """Map 0..1 intensities to a smooth teal→amber attention palette."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(0.05 + 0.90 * v, 0.0, 1.0)
    g = np.clip(0.55 + 0.25 * (1.0 - v), 0.0, 1.0)
    b = np.clip(0.55 * (1.0 - v) + 0.08 * v, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _retinal_fov_mask(rgb: np.ndarray) -> np.ndarray:
    """Binary-ish FOV mask from luminance so overlays stay inside the fundus disk."""
    luminance = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).astype(np.float32)
    threshold = max(8.0, float(np.percentile(luminance, 5)) * 0.6)
    mask = (luminance > threshold).astype(np.float32)
    # Center the prior on the same fundus crop as inference, including scans
    # with asymmetric black borders.
    from retinal_preprocessing import retina_bounds
    x, y, crop_w, crop_h = retina_bounds(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w = luminance.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = y + (crop_h - 1) / 2.0, x + (crop_w - 1) / 2.0
    radius = 0.48 * min(crop_h, crop_w)
    circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)
    mask = mask * circle.astype(np.float32)
    mask_img = Image.fromarray(np.uint8(np.clip(mask * 255.0, 0, 255)))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def simulate_progression(
    image_path: str,
    gradcam_path: str,
    progression_history,
    random_seed: int | None = None,
) -> str:
    """Create a deterministic illustrative attention overlay.

    ``random_seed`` is accepted for API compatibility but unused — the
    visualization is fully deterministic given the same inputs.
    """
    del random_seed  # deterministic by design

    image_file = Path(image_path)
    heatmap_file = Path(gradcam_path)

    if not image_file.exists():
        raise FileNotFoundError(f"Retinal image not found: {image_path}")
    if not heatmap_file.exists():
        raise FileNotFoundError(f"Grad-CAM file not found: {gradcam_path}")
    if not progression_history:
        raise ValueError("Progression history is empty; cannot build illustrative overlay.")

    with Image.open(image_file) as source:
        original = source.convert("RGB")
    with Image.open(heatmap_file) as source:
        if source.mode != "L":
            raise ValueError("Simulation requires genuine scalar Grad-CAM attention, not an RGB overlay")
        heatmap = source.copy()

    width, height = original.size
    heatmap = heatmap.resize((width, height), Image.Resampling.BILINEAR)

    base = np.asarray(original).astype(np.float32) / 255.0
    fov = _retinal_fov_mask(base * 255.0)

    heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0
    mask = heatmap_array * fov
    if float(mask.max()) <= 0:
        raise ValueError("No model attention is available inside the retinal field of view")

    last_status = progression_history[-1].get("status") if progression_history else "Initial"
    spread_factor = _status_spread_factor(str(last_status or "Initial"))

    mask_img = Image.fromarray(np.uint8(np.clip(mask * 255.0, 0, 255)))
    if spread_factor > 1.0:
        grow = 1 + int(round((spread_factor - 1.0) * 8))
        grow = max(3, grow | 1)  # odd kernel
        mask_img = mask_img.filter(ImageFilter.MaxFilter(grow))
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        mask = np.clip(mask * (0.85 + 0.15 * spread_factor), 0.0, 1.0)
    elif spread_factor < 1.0:
        shrink = 5
        mask_img = mask_img.filter(ImageFilter.MinFilter(shrink))
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        mask = np.clip(mask * spread_factor, 0.0, 1.0)

    mask = np.asarray(
        Image.fromarray(np.uint8(np.clip(mask * 255.0, 0, 255))).filter(
            ImageFilter.GaussianBlur(radius=2.0)
        ),
        dtype=np.float32,
    ) / 255.0
    mask = np.clip(mask * fov, 0.0, 1.0)

    heat_rgb = _attention_rgb(mask)
    alpha = (0.12 + 0.48 * mask)[..., None] * fov[..., None]
    blended = np.clip(base * (1.0 - alpha) + heat_rgb * alpha, 0.0, 1.0)

    # Preserve vessel/disc visibility by limiting overlay strength on bright structures.
    luminance = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    preserve = np.clip((luminance - 0.55) / 0.35, 0.0, 1.0)[..., None]
    blended = blended * (1.0 - 0.35 * preserve) + base * (0.35 * preserve)

    result_image = Image.fromarray(np.uint8(np.clip(blended * 255.0, 0, 255)))
    draw = ImageDraw.Draw(result_image, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bar_h = 30
    draw.rectangle((0, height - bar_h, width, height), fill=(15, 23, 42, 180))
    draw.text((12, height - 22), ILLUSTRATIVE_CAPTION, fill=(248, 250, 252, 240), font=font)

    output_dir = Path(config.SIMULATION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_file.stem}_simulation_v2.png"
    result_image.save(output_path)
    return str(output_path)


def simulation_method_description() -> str:
    return METHOD_DESCRIPTION
