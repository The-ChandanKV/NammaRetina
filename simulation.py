from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def simulate_progression(
    image_path: str,
    gradcam_path: str,
    progression_history,
    random_seed: int | None = None,
) -> str:
    """Create a simple probabilistic visualization of a possible future affected region.

    This is not a clinically validated disease forecast. It is a lightweight visualization
    based on the original retinal image, a Grad-CAM heatmap, and the patient's progression trend.
    """
    image_file = Path(image_path)
    heatmap_file = Path(gradcam_path)

    if not image_file.exists():
        raise FileNotFoundError(f"Retinal image not found: {image_path}")
    if not heatmap_file.exists():
        raise FileNotFoundError(f"Grad-CAM file not found: {gradcam_path}")
    if not progression_history:
        raise ValueError("Progression history is empty; cannot simulate future spread.")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    original = Image.open(image_file).convert("RGB")
    heatmap = Image.open(heatmap_file).convert("L")

    width, height = original.size
    heatmap = heatmap.resize((width, height))

    heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0
    mask = (heatmap_array > 0.30).astype(np.float32)
    if not np.any(mask):
        mask = np.ones_like(heatmap_array, dtype=np.float32)

    last_status = progression_history[-1]["status"] if progression_history else "Initial"
    if last_status == "Worsened":
        spread_factor = 1.35
    elif last_status == "Stable":
        spread_factor = 1.10
    elif last_status == "Improved":
        spread_factor = 0.85
    else:
        spread_factor = 1.00

    if spread_factor > 1.0:
        mask = np.clip(mask * spread_factor, 0.0, 1.0)
        mask = np.asarray(Image.fromarray(mask * 255).filter(ImageFilter.MaxFilter(5))) / 255.0
    elif spread_factor < 1.0:
        mask = np.clip(mask * spread_factor, 0.0, 1.0)
        mask = np.asarray(Image.fromarray(mask * 255).filter(ImageFilter.MinFilter(5))) / 255.0

    noise = np.random.uniform(0.0, 0.25, size=mask.shape)
    probability_map = np.clip(mask + noise, 0.0, 1.0)

    overlay = np.asarray(original).astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] + 40 * probability_map, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] - 30 * probability_map, 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] - 20 * probability_map, 0, 255)

    output_dir = Path(__file__).resolve().parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulated_progression.png"

    result_image = Image.fromarray(np.uint8(overlay))
    result_image.save(output_path)
    return str(output_path)
