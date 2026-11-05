"""Load evaluation run artifacts for the administrator research dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "evaluation_runs"
LEGACY_METRICS = Path(__file__).resolve().parent / "training" / "results" / "cpu_metrics.json"


def list_evaluation_runs(output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> list[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def load_run(run_dir: Path | str) -> Optional[dict]:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("status") != "complete":
            return None
        if not all(isinstance(result.get(key), dict) for key in
                   ("dataset", "model", "five_class", "referable_dr", "severe_dr", "calibration")):
            return None
        # A syntactically valid but incomplete file must not crash formatting.
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "quadratic_weighted_kappa"):
            value = result["five_class"].get(key)
            if key not in result["five_class"] or (value is not None and not isinstance(value, (int, float))):
                return None
        return result
    except (OSError, ValueError):
        return None


def latest_evaluation(output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> Optional[dict]:
    runs = list_evaluation_runs(output_root)
    for run in runs:
        payload = load_run(run)
        if payload:
            payload["run_id"] = run.name
            payload["run_path"] = str(run)
            return payload
    return None


def ensure_seed_evaluation(output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> Optional[dict]:
    """If no runs exist, import legacy training metrics when available (no fabrication)."""
    existing = latest_evaluation(output_root)
    if existing:
        return existing
    if not LEGACY_METRICS.exists():
        return None
    from model_evaluation import import_legacy_training_metrics, write_evaluation_run

    result = import_legacy_training_metrics(LEGACY_METRICS, dataset_name="aptos_internal_validation")
    result["external_validation"] = {
        "status": "pending",
        "message": "External validation pending.",
        "note": (
            "EyePACS images may exist under training/data/eyepacs224, but an independent "
            "scored evaluation run has not been written for this dashboard yet."
        ),
    }
    write_evaluation_run(result, output_root)
    return latest_evaluation(output_root)
