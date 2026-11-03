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
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def latest_evaluation(output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> Optional[dict]:
    runs = list_evaluation_runs(output_root)
    if not runs:
        return None
    payload = load_run(runs[0])
    if payload:
        payload["run_id"] = runs[0].name
        payload["run_path"] = str(runs[0])
    return payload


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
