from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config


def _status_spread_factor(status: str) -> float:
    if status == "Worsened":
        return 1.40
    if status == "Stable":
        return 1.12
    if status == "Improved":
        return 0.82
    return 1.05  # Initial / unknown — mild illustrative expansion from Grad-CAM


def _jet_rgb(values: np.ndarray) -> np.ndarray:
    """Map 0..1 intensities to a smooth yellow→orange→red attention palette."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(0.55 + 0.45 * v, 0.0, 1.0)
    g = np.clip(0.75 * (1.0 - v) + 0.15 * v, 0.0, 1.0)
    b = np.clip(0.12 * (1.0 - v), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def simulate_progression(
    image_path: str,
    gradcam_path: str,
    progression_history,
    random_seed: int | None = None,
) -> str:
    """Create a probabilistic visualization of a possible future affected region.

    This is not a clinically validated disease forecast. It blends the original
    retinal image with a Grad-CAM-derived attention mask modulated by the
    patient's recorded progression trend.
    """
    image_file = Path(image_path)
    heatmap_file = Path(gradcam_path)

    if not image_file.exists():
        raise FileNotFoundError(f"Retinal image not found: {image_path}")
    if not heatmap_file.exists():
        raise FileNotFoundError(f"Grad-CAM file not found: {gradcam_path}")
    if not progression_history:
        raise ValueError("Progression history is empty; cannot simulate future spread.")

    if random_seed is None:
        digest = hashlib.sha256(image_file.name.encode("utf-8")).hexdigest()
        random_seed = int(digest[:8], 16)
    rng = np.random.default_rng(random_seed)

    original = Image.open(image_file).convert("RGB")
    heatmap = Image.open(heatmap_file).convert("L")

    width, height = original.size
    heatmap = heatmap.resize((width, height), Image.Resampling.BILINEAR)

    heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0
    # Prefer warm Grad-CAM cores; fall back to full field if empty.
    mask = np.clip((heatmap_array - 0.22) / 0.78, 0.0, 1.0)
    if float(mask.max()) < 0.05:
        yy, xx = np.mgrid[0:height, 0:width]
        cy, cx = height / 2.0, width / 2.0
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = np.clip(1.0 - dist / (0.42 * max(width, height)), 0.0, 1.0)

    last_status = progression_history[-1].get("status") if progression_history else "Initial"
    spread_factor = _status_spread_factor(str(last_status or "Initial"))

    mask_img = Image.fromarray(np.uint8(np.clip(mask * 255.0, 0, 255)))
    if spread_factor > 1.0:
        grow = 1 + int(round((spread_factor - 1.0) * 8))
        grow = max(3, grow | 1)  # odd kernel
        mask_img = mask_img.filter(ImageFilter.MaxFilter(grow))
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        mask = np.clip(mask * spread_factor, 0.0, 1.0)
    elif spread_factor < 1.0:
        shrink = 5
        mask_img = mask_img.filter(ImageFilter.MinFilter(shrink))
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        mask = np.clip(mask * spread_factor, 0.0, 1.0)

    mask = np.asarray(
        Image.fromarray(np.uint8(np.clip(mask * 255.0, 0, 255))).filter(ImageFilter.GaussianBlur(radius=2.2)),
        dtype=np.float32,
    ) / 255.0

    noise = rng.uniform(0.0, 0.18, size=mask.shape).astype(np.float32)
    probability_map = np.clip(mask * 0.88 + noise * mask, 0.0, 1.0)

    base = np.asarray(original).astype(np.float32) / 255.0
    heat_rgb = _jet_rgb(probability_map)
    alpha = (0.15 + 0.55 * probability_map)[..., None]
    blended = np.clip(base * (1.0 - alpha) + heat_rgb * alpha, 0.0, 1.0)

    result_image = Image.fromarray(np.uint8(blended * 255.0))
    draw = ImageDraw.Draw(result_image, "RGBA")
    # Soft ring cue around high-probability cores (visual only).
    hot = probability_map > 0.62
    if np.any(hot):
        ys, xs = np.where(hot)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad = max(8, int(0.02 * max(width, height)))
        draw.ellipse(
            (x0 - pad, y0 - pad, x1 + pad, y1 + pad),
            outline=(255, 210, 80, 140),
            width=max(2, width // 280),
        )

    caption = "Illustrative spread simulation — not a clinical forecast"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle((8, height - 28, width - 8, height - 8), fill=(7, 20, 33, 150))
    draw.text((14, height - 24), caption, fill=(235, 245, 255, 230), font=font)

    output_dir = Path(config.SIMULATION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_file.stem}_simulation.png"
    result_image.save(output_path)
    return str(output_path)
