"""Technical diagnostics for Grad-CAM and LIME explanations.

These checks measure reproducibility and surrogate fidelity. They do not prove
clinical meaningfulness of highlighted regions.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from lime_explainer import explain_image_lime


def lime_surrogate_fidelity(
    image_path: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_class: int,
    **kwargs,
) -> dict:
    result = explain_image_lime(image_path, predict_fn, target_class, **kwargs)
    return {
        "fidelity_r2": result["fidelity_r2"],
        "n_samples": result["n_samples"],
        "n_segments": result["n_segments"],
        "elapsed_ms": result["elapsed_ms"],
        "interpretation": (
            "R² of the local Ridge surrogate vs black-box scores on weighted "
            "perturbations. High fidelity does not imply clinical correctness."
        ),
    }


def lime_stability(
    image_path: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_class: int,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    top_k: int = 5,
    **kwargs,
) -> dict:
    """Jaccard overlap of top-|weight| superpixels across seeds."""
    sets = []
    fidelities = []
    for seed in seeds:
        result = explain_image_lime(
            image_path, predict_fn, target_class, seed=seed, top_k=top_k, **kwargs
        )
        top = {row["segment"] for row in result["top_regions"][:top_k]}
        sets.append(top)
        fidelities.append(result["fidelity_r2"])
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            pairs.append(len(inter) / len(union) if union else 1.0)
    return {
        "mean_top_jaccard": float(np.mean(pairs)) if pairs else None,
        "pair_jaccards": pairs,
        "mean_fidelity_r2": float(np.mean(fidelities)) if fidelities else None,
        "seeds": list(seeds),
        "note": "Stability under seed changes is a research diagnostic only.",
    }


def gradcam_reproducibility(map_a: np.ndarray, map_b: np.ndarray) -> dict:
    """Compare two Grad-CAM maps after min-max normalization."""
    a = np.asarray(map_a, dtype=np.float64)
    b = np.asarray(map_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Grad-CAM maps must share shape")
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Grad-CAM maps must be nonempty and finite")
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else None
    mae = float(np.mean(np.abs(a - b)))
    return {
        "pearson_r": corr,
        "mae": mae,
        "aligned_shapes": True,
        "note": "Reproducibility of maps under identical inputs; not a clinical claim.",
    }


def deletion_auc(
    image_rgb_uint8: np.ndarray,
    importance_map: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_class: int,
    *,
    steps: int = 10,
    hide_value: float = 0.0,
) -> dict:
    """Simple deletion curve AUC (research diagnostic).

    Progressively replaces highest-importance pixels and records target-class
    probability drop. Lower remaining probability faster ⇒ higher attribution
    faithfulness under this proxy.
    """
    image = np.asarray(image_rgb_uint8, dtype=np.float32)
    importance = np.asarray(importance_map, dtype=np.float64)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if importance.ndim != 2 or importance.shape != image.shape[:2]:
        raise ValueError("importance map must match image spatial size")
    flat_order = np.argsort(-importance.ravel())
    n = flat_order.size
    probs = []
    working = image.copy()
    for step in range(steps + 1):
        batch = working[None, ...] / 127.5 - 1.0
        pred = np.asarray(predict_fn(batch), dtype=np.float64)[0, target_class]
        probs.append(float(pred))
        if step == steps:
            break
        start = step * n // steps
        end = (step + 1) * n // steps
        idx = flat_order[start:end]
        ys, xs = np.unravel_index(idx, importance.shape[:2])
        working[ys, xs] = hide_value
    # Normalize deletion curve area (lower is more faithful under deletion).
    xs = np.linspace(0.0, 1.0, len(probs))
    auc = float(np.sum((np.asarray(probs[:-1]) + probs[1:]) * np.diff(xs) / 2))
    return {
        "deletion_auc": auc,
        "curve": probs,
        "steps": steps,
        "interpretation": (
            "Research proxy only. Do not equate deletion AUC with clinical validity."
        ),
    }
