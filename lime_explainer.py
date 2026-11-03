"""Genuine LIME-style explanations for retinal image classifiers.

Perturbs superpixel masks, scores the black-box model on each perturbation,
and fits a local weighted linear surrogate. Grad-CAM intensities are never used
as a substitute for the surrogate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

PredictFn = Callable[[np.ndarray], np.ndarray]  # (N,H,W,3) float32 in model space → (N,C)


def _slic_superpixels(image_rgb_uint8: np.ndarray, n_segments: int = 40, compactness: float = 10.0) -> np.ndarray:
    """Compact SLIC-like segmentation using Lab color + spatial clustering."""
    try:
        from skimage.segmentation import slic

        return slic(
            image_rgb_uint8,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        ).astype(np.int32)
    except Exception:
        # Fallback: regular grid (still used as interpretable features for LIME).
        h, w = image_rgb_uint8.shape[:2]
        rows = max(int(np.sqrt(n_segments * h / w)), 1)
        cols = max(int(np.sqrt(n_segments * w / h)), 1)
        labels = np.zeros((h, w), dtype=np.int32)
        seg = 0
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * h // rows, (r + 1) * h // rows
                x0, x1 = c * w // cols, (c + 1) * w // cols
                labels[y0:y1, x0:x1] = seg
                seg += 1
        return labels


def _to_model_batch(images_rgb_uint8: np.ndarray) -> np.ndarray:
    """Match ``model_loader.preprocess_image`` normalization: RGB /127.5 - 1."""
    return images_rgb_uint8.astype(np.float32) / 127.5 - 1.0


def _prepare_display_image(image_path: str, size: int) -> np.ndarray:
    from retinal_preprocessing import prepare_retina

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    prepared = prepare_retina(bgr, size)
    return cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)


def explain_image_lime(
    image_path: str,
    predict_fn: PredictFn,
    target_class: int,
    *,
    n_samples: int = 200,
    n_segments: int = 40,
    batch_size: int = 16,
    seed: int = 42,
    image_size: int = 224,
    hide_color: Optional[np.ndarray] = None,
    top_k: int = 8,
) -> dict:
    """Run genuine LIME for ``target_class`` on a retinal image.

    Parameters
    ----------
    predict_fn:
        Callable that accepts a batch of preprocessed images shaped
        ``(N, H, W, 3)`` in the model input space and returns class
        probabilities ``(N, C)``.
    hide_color:
        Baseline fill for masked superpixels (default: image mean).
    """
    started = time.perf_counter()
    rgb = _prepare_display_image(image_path, image_size)
    labels = _slic_superpixels(rgb, n_segments=n_segments)
    segment_ids = np.unique(labels)
    n_features = int(len(segment_ids))
    if n_features < 2:
        raise ValueError("LIME requires at least two superpixels")

    if hide_color is None:
        hide_color = rgb.mean(axis=(0, 1))
    hide_color = np.asarray(hide_color, dtype=np.float32).reshape(1, 1, 3)

    rng = np.random.default_rng(seed)
    # Include the original (all segments present) as the first sample.
    presence = np.ones((n_samples, n_features), dtype=np.float32)
    if n_samples > 1:
        presence[1:] = rng.integers(0, 2, size=(n_samples - 1, n_features)).astype(np.float32)
        # Avoid empty images.
        empty = presence.sum(axis=1) == 0
        presence[empty, rng.integers(0, n_features, size=int(empty.sum()))] = 1.0

    id_to_col = {int(seg): i for i, seg in enumerate(segment_ids)}
    probabilities = np.zeros(n_samples, dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_imgs = []
        for row in presence[start:end]:
            mask = np.ones(rgb.shape[:2], dtype=np.float32)
            for seg, col in id_to_col.items():
                if row[col] < 0.5:
                    mask[labels == seg] = 0.0
            perturbed = rgb.astype(np.float32) * mask[..., None] + hide_color * (1.0 - mask[..., None])
            batch_imgs.append(perturbed)
        batch = _to_model_batch(np.stack(batch_imgs, axis=0))
        preds = np.asarray(predict_fn(batch), dtype=np.float64)
        if preds.ndim != 2 or preds.shape[0] != (end - start):
            raise ValueError("predict_fn returned an unexpected shape")
        if target_class < 0 or target_class >= preds.shape[1]:
            raise ValueError("target_class out of range for model outputs")
        probabilities[start:end] = preds[:, target_class]

    # Proximity kernel in binary presence space (cosine / Hamming affinity).
    distances = np.abs(presence - presence[0:1]).sum(axis=1) / n_features
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))

    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(presence, probabilities, sample_weight=weights)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = float(model.intercept_)
    pred_local = model.predict(presence)
    ss_res = float(np.sum(weights * (probabilities - pred_local) ** 2))
    ss_tot = float(np.sum(weights * (probabilities - np.average(probabilities, weights=weights)) ** 2))
    fidelity_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    order = np.argsort(-np.abs(coef))
    top_idx = order[: max(1, min(top_k, n_features))]
    positive = [(int(segment_ids[i]), float(coef[i])) for i in top_idx if coef[i] > 0]
    negative = [(int(segment_ids[i]), float(coef[i])) for i in top_idx if coef[i] < 0]

    overlay = rgb.copy()
    pos_mask = np.zeros(labels.shape, dtype=bool)
    neg_mask = np.zeros(labels.shape, dtype=bool)
    for seg, weight in positive:
        pos_mask |= labels == seg
    for seg, weight in negative:
        neg_mask |= labels == seg

    overlay = overlay.astype(np.float32)
    if pos_mask.any():
        overlay[pos_mask] = overlay[pos_mask] * 0.55 + np.array([40, 180, 120], dtype=np.float32) * 0.45
    if neg_mask.any():
        overlay[neg_mask] = overlay[neg_mask] * 0.55 + np.array([220, 80, 80], dtype=np.float32) * 0.45
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    for mask, color in ((pos_mask, (0, 160, 120)), (neg_mask, (200, 60, 60))):
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_u8, contours, -1, color, 1)

    boundary = overlay_u8.copy()
    for seg in segment_ids:
        edge = (labels == seg).astype(np.uint8) * 255
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, (210, 210, 210), 1)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    top_regions = []
    for rank, i in enumerate(top_idx, start=1):
        seg = int(segment_ids[i])
        top_regions.append(
            {
                "rank": rank,
                "segment": seg,
                "weight": round(float(coef[i]), 4),
                "direction": "positive" if coef[i] >= 0 else "negative",
                "area_pct": round(float(np.mean(labels == seg) * 100.0), 2),
            }
        )

    return {
        "method": "lime_superpixel_ridge",
        "target_class": int(target_class),
        "n_samples": int(n_samples),
        "n_segments": n_features,
        "batch_size": int(batch_size),
        "seed": int(seed),
        "image_size": int(image_size),
        "fidelity_r2": round(fidelity_r2, 4),
        "intercept": round(intercept, 6),
        "coefficients": {str(int(segment_ids[i])): float(coef[i]) for i in range(n_features)},
        "top_regions": top_regions,
        "positive_segments": positive,
        "negative_segments": negative,
        "overlay_rgb": overlay_u8,
        "boundary_rgb": boundary,
        "elapsed_ms": round(elapsed_ms, 1),
        "disclaimer": (
            "LIME estimates local influence of superpixels under perturbation. "
            "Highlighted regions are not confirmed pathological lesions."
        ),
    }


def explain_with_loaded_model(
    image_path: str,
    target_class: int,
    **kwargs,
) -> dict:
    """Convenience wrapper using the singleton EfficientNetB0 classifier."""
    from model_loader import load_model

    model = load_model()
    if model is None:
        raise RuntimeError("Trained model is unavailable for LIME explanations")

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(batch, verbose=0))

    return explain_image_lime(image_path, predict_fn, target_class, **kwargs)


def save_lime_artifact(result: dict, output_dir: str | Path, report_id: int) -> dict:
    """Persist overlay PNG + metadata JSON for caching."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overlay_path = out / f"lime_{report_id}.png"
    meta_path = out / f"lime_{report_id}.json"
    bgr = cv2.cvtColor(result["overlay_rgb"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), bgr)
    serializable = {
        k: v
        for k, v in result.items()
        if k not in {"overlay_rgb", "boundary_rgb"}
    }
    serializable["overlay_path"] = str(overlay_path)
    digest = hashlib.sha256(json.dumps(serializable, sort_keys=True, default=str).encode()).hexdigest()[:16]
    serializable["cache_key"] = digest
    meta_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return serializable


def load_lime_artifact(output_dir: str | Path, report_id: int) -> Optional[dict]:
    meta_path = Path(output_dir) / f"lime_{report_id}.json"
    overlay_path = Path(output_dir) / f"lime_{report_id}.png"
    if not meta_path.exists() or not overlay_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["overlay_path"] = str(overlay_path)
    return data
