"""Clinical research evaluation framework for the five-class DR classifier.

This is a research evaluation module — not regulatory or clinical approval.
Never fabricates metrics: if labels/predictions are missing, status is pending.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "evaluation_runs"


def _as_probs(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("probabilities must have shape (n, 5)")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < -1e-6) or np.any(probabilities > 1 + 1e-6):
        raise ValueError("probabilities must be in [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.02):
        raise ValueError("each probability row must sum to ~1")
    return probabilities


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    patient_ids: Optional[np.ndarray] = None,
) -> list[float]:
    """95% CI via bootstrap; patient-level when patient_ids are provided."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = []
    if patient_ids is None:
        n = len(values)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            samples.append(float(statistic(values[idx])))
    else:
        patient_ids = np.asarray(patient_ids)
        unique = np.unique(patient_ids)
        for _ in range(n_boot):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            mask = np.isin(patient_ids, chosen)
            if not np.any(mask):
                continue
            samples.append(float(statistic(values[mask])))
    if not samples:
        return [float("nan"), float("nan")]
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    f1 = (2 * prec * sens / (prec + sens)) if (prec is not None and sens is not None and (prec + sens)) else None
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": npv,
        "f1": f1,
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(np.sum(~y_true)),
    }


def expected_calibration_error(y_true: np.ndarray, confidences: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_diagram(y_true: np.ndarray, confidences: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if not np.any(mask):
            rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0, "accuracy": None, "confidence": None})
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": int(mask.sum()),
                "accuracy": float(np.mean(y_true[mask])),
                "confidence": float(np.mean(confidences[mask])),
            }
        )
    return rows


def evaluate_predictions(
    y_true,
    probabilities,
    *,
    patient_ids=None,
    dataset_name: str = "unspecified",
    split_name: str = "test",
    model_checkpoint: str = "",
    model_sha256: str = "",
    referable_threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Compute five-class, referable-DR, severe-DR, and calibration metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = _as_probs(probabilities)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities length mismatch")
    if np.any(y_true < 0) or np.any(y_true > 4):
        raise ValueError("y_true must be in 0..4")

    predicted = probabilities.argmax(axis=1)
    precision_m, recall_m, f1_m, support = precision_recall_fscore_support(
        y_true, predicted, labels=range(5), average=None, zero_division=0
    )

    five_class = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, predicted, weights="quadratic", labels=range(5))
        ),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision_m[i]),
                "recall": float(recall_m[i]),
                "f1": float(f1_m[i]),
                "support": int(support[i]),
            }
            for i in range(5)
        },
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=range(5)).tolist(),
        "classification_report": classification_report(
            y_true, predicted, labels=range(5), target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }

    # Referable DR: grade ≥ 2 (moderate or worse). Score = P(class≥2).
    y_ref = y_true >= 2
    score_ref = probabilities[:, 2:].sum(axis=1)
    pred_ref = score_ref >= referable_threshold
    referable = binary_confusion(y_ref, pred_ref)
    referable.update(
        {
            "definition": "Referable DR = grade 2–4 (moderate or worse)",
            "score_threshold": referable_threshold,
            "roc_auc": float(roc_auc_score(y_ref, score_ref)) if y_ref.any() and (~y_ref).any() else None,
            "pr_auc": float(average_precision_score(y_ref, score_ref)) if y_ref.any() else None,
        }
    )

    def _sens_from_pairs(rows):
        return binary_confusion(rows[:, 0].astype(bool), rows[:, 1].astype(bool))["sensitivity"] or 0.0

    def _spec_from_pairs(rows):
        return binary_confusion(rows[:, 0].astype(bool), rows[:, 1].astype(bool))["specificity"] or 0.0

    pairs_ref = np.column_stack([y_ref.astype(float), pred_ref.astype(float)])
    referable["sensitivity_ci95"] = bootstrap_ci(pairs_ref, _sens_from_pairs, seed=seed, patient_ids=patient_ids)
    referable["specificity_ci95"] = bootstrap_ci(pairs_ref, _spec_from_pairs, seed=seed + 1, patient_ids=patient_ids)

    # Severe / proliferative: grade ≥ 3.
    y_sev = y_true >= 3
    score_sev = probabilities[:, 3:].sum(axis=1)
    pred_sev = score_sev >= referable_threshold
    severe = binary_confusion(y_sev, pred_sev)
    severe.update(
        {
            "definition": "Severe-or-worse DR = grade 3–4",
            "score_threshold": referable_threshold,
            "roc_auc": float(roc_auc_score(y_sev, score_sev)) if y_sev.any() and (~y_sev).any() else None,
            "pr_auc": float(average_precision_score(y_sev, score_sev)) if y_sev.any() else None,
        }
    )
    pairs_sev = np.column_stack([y_sev.astype(float), pred_sev.astype(float)])
    severe["sensitivity_ci95"] = bootstrap_ci(pairs_sev, _sens_from_pairs, seed=seed + 2, patient_ids=patient_ids)
    severe["specificity_ci95"] = bootstrap_ci(pairs_sev, _spec_from_pairs, seed=seed + 3, patient_ids=patient_ids)

    confidences = probabilities.max(axis=1)
    correct = (predicted == y_true).astype(int)
    calibration = {
        "brier_score_multiclass": float(np.mean(np.sum((np.eye(5)[y_true] - probabilities) ** 2, axis=1))),
        "brier_score_top_class": float(brier_score_loss(correct, confidences)),
        "expected_calibration_error": expected_calibration_error(correct, confidences),
        "reliability_diagram": reliability_diagram(correct, confidences),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "note": "Temperature scaling is not fitted on this evaluation split.",
    }

    accuracy_ci = bootstrap_ci(
        (predicted == y_true).astype(float),
        np.mean,
        seed=seed + 4,
        patient_ids=patient_ids,
    )

    return {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "split": split_name,
            "n": int(len(y_true)),
            "class_names": CLASS_NAMES,
            "patient_level_bootstrap": patient_ids is not None,
        },
        "model": {
            "checkpoint": model_checkpoint,
            "sha256": model_sha256,
        },
        "five_class": five_class,
        "accuracy_ci95": accuracy_ci,
        "referable_dr": referable,
        "severe_dr": severe,
        "calibration": calibration,
        "limitations": [
            "Research evaluation only — not clinical or regulatory validation.",
            "Do not report training-set performance as independent test performance.",
            "APTOS lacks patient IDs; image-level isolation is used unless IDs are supplied.",
            "External validation requires non-overlapping held-out datasets.",
        ],
        "external_validation": {
            "status": "pending",
            "message": "External validation pending.",
        },
    }


def write_evaluation_run(result: dict, output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> Path:
    root = Path(output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset = result.get("dataset", {}).get("name", "dataset")
    split = result.get("dataset", {}).get("split", "split")
    run_dir = root / f"{stamp}_{dataset}_{split}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Flatten CSV for five-class per-class rows.
    lines = ["class,precision,recall,f1,support"]
    for name, row in result["five_class"]["per_class"].items():
        lines.append(f"{name},{row['precision']},{row['recall']},{row['f1']},{row['support']}")
    (run_dir / "per_class_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    five = result.get("five_class") or {}
    referable = result.get("referable_dr") or {}
    severe = result.get("severe_dr") or {}
    calibration = result.get("calibration") or {}
    ece = calibration.get("expected_calibration_error")
    brier = calibration.get("brier_score_multiclass")
    report = [
        f"# Evaluation report — {dataset} / {split}",
        "",
        f"Generated: {result['generated_at']}",
        f"Samples: {result['dataset']['n']}",
        f"Model checkpoint: {result['model'].get('checkpoint') or 'unspecified'}",
        f"Model SHA256: {result['model'].get('sha256') or 'unspecified'}",
        "",
        "## Five-class",
        f"- Accuracy: {five.get('accuracy')} (95% CI {result.get('accuracy_ci95')})",
        f"- Balanced accuracy: {five.get('balanced_accuracy')}",
        f"- Macro F1: {five.get('macro_f1')}",
        f"- Quadratic weighted kappa: {five.get('quadratic_weighted_kappa')}",
        "",
        "## Referable DR (grade ≥ 2)",
        f"- Sensitivity: {referable.get('sensitivity')}",
        f"- Specificity: {referable.get('specificity')}",
        f"- ROC-AUC: {referable.get('roc_auc')}",
        "",
        "## Severe DR (grade ≥ 3)",
        f"- Sensitivity: {severe.get('sensitivity')}",
        f"- Specificity: {severe.get('specificity')}",
        "",
        "## Calibration",
        f"- ECE: {ece if ece is not None else calibration.get('message') or 'pending'}",
        f"- Multiclass Brier: {brier if brier is not None else 'pending'}",
        "",
        "## Limitations",
        *[f"- {item}" for item in result.get("limitations") or []],
        "",
        f"External validation: {(result.get('external_validation') or {}).get('message')}",
        "",
        "Clinical Limitations and Research Status: technical metrics do not establish",
        "clinical effectiveness or regulatory approval.",
    ]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return run_dir


def load_predictions_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Load ``y_true``, ``probabilities``, optional ``patient_ids`` from NPZ/JSON."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        y_true = data["y_true"]
        probabilities = data["probabilities"]
        patient_ids = data["patient_ids"] if "patient_ids" in data.files else None
        meta = {k: data[k].item() if data[k].shape == () else data[k].tolist()
                for k in data.files if k not in {"y_true", "probabilities", "patient_ids"}}
        return y_true, probabilities, patient_ids, meta
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    y_true = np.asarray(payload["y_true"])
    probabilities = np.asarray(payload["probabilities"])
    patient_ids = np.asarray(payload["patient_ids"]) if "patient_ids" in payload else None
    meta = {k: v for k, v in payload.items() if k not in {"y_true", "probabilities", "patient_ids"}}
    return y_true, probabilities, patient_ids, meta


def import_legacy_training_metrics(path: Path | str, dataset_name: str = "aptos_validation_legacy") -> dict:
    """Wrap an existing training/results metrics JSON into the dashboard schema."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Prefer validation block when present.
    block = raw.get("validation") or raw.get("test") or raw
    if "accuracy" not in block:
        raise ValueError(f"Unrecognized metrics file: {path}")
    cm = block.get("confusion_matrix")
    per_class_recall = block.get("per_class_recall") or []
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": None,
            "recall": float(per_class_recall[i]) if i < len(per_class_recall) else None,
            "f1": None,
            "support": None,
        }
    return {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(path),
        "dataset": {
            "name": dataset_name,
            "split": "validation",
            "n": int(block.get("n") or 0),
            "class_names": CLASS_NAMES,
            "patient_level_bootstrap": False,
        },
        "model": {"checkpoint": "legacy_training_results", "sha256": ""},
        "five_class": {
            "n": int(block.get("n") or 0),
            "accuracy": float(block["accuracy"]),
            "balanced_accuracy": float(block.get("balanced_accuracy") or 0),
            "macro_precision": None,
            "macro_recall": float(np.mean(per_class_recall)) if per_class_recall else None,
            "macro_f1": float(block.get("macro_f1") or 0),
            "weighted_f1": None,
            "quadratic_weighted_kappa": float(block.get("quadratic_weighted_kappa") or 0),
            "per_class": per_class,
            "confusion_matrix": cm,
            "classification_report": block.get("classification_report"),
        },
        "accuracy_ci95": None,
        "referable_dr": {
            "definition": "Referable DR = grade 2–4 (moderate or worse)",
            "score_threshold": block.get("referable_dr_threshold", 0.5),
            "sensitivity": block.get("referable_dr_sensitivity"),
            "specificity": block.get("referable_dr_specificity"),
            "precision": None,
            "npv": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "sensitivity_ci95": None,
            "specificity_ci95": None,
            "tp": None,
            "tn": None,
            "fp": None,
            "fn": None,
        },
        "severe_dr": {
            "definition": "Severe-or-worse DR = grade 3–4",
            "status": "partial",
            "note": "Legacy file did not store a full severe-DR confusion matrix.",
            "sensitivity": None,
            "specificity": None,
        },
        "calibration": {
            "status": "pending",
            "message": "No validated evaluation results available for calibration on this legacy file.",
        },
        "limitations": [
            "Imported from a prior training metrics file; not all research metrics were stored.",
            "Research evaluation only — not clinical or regulatory validation.",
        ],
        "external_validation": {
            "status": "pending",
            "message": "External validation pending.",
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NammaRetina clinical evaluation runner")
    parser.add_argument("--predictions", help="NPZ or JSON with y_true and probabilities")
    parser.add_argument("--import-legacy", help="Import training/results/*.json metrics")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--sha256", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)

    if args.import_legacy:
        result = import_legacy_training_metrics(args.import_legacy, dataset_name=args.dataset)
    elif args.predictions:
        y_true, probabilities, patient_ids, meta = load_predictions_npz(args.predictions)
        result = evaluate_predictions(
            y_true,
            probabilities,
            patient_ids=patient_ids,
            dataset_name=args.dataset or meta.get("dataset_name", "custom"),
            split_name=args.split or meta.get("split", "test"),
            model_checkpoint=args.checkpoint or meta.get("checkpoint", ""),
            model_sha256=args.sha256 or meta.get("sha256", ""),
        )
    else:
        parser.error("Provide --predictions or --import-legacy")
        return 2

    run_dir = write_evaluation_run(result, args.output_root)
    print(f"Wrote evaluation run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
