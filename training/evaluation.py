"""Metrics for five-grade DR; no thresholds are fitted using test labels."""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, recall_score)

from training.data_pipeline import CLASS_NAMES


def metrics(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(y_true), 5) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Invalid prediction array")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=0.01):
        raise ValueError("Invalid class probabilities")
    predicted = probabilities.argmax(axis=1)
    severity_mask = y_true >= 3
    referable = probabilities[:, 2:].sum(axis=1) >= 0.5
    positives = y_true >= 2
    tp = int(np.sum(positives & referable))
    fn = int(np.sum(positives & ~referable))
    tn = int(np.sum(~positives & ~referable))
    fp = int(np.sum(~positives & referable))
    return {
        "n": len(y_true), "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=range(5), average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, predicted, weights="quadratic", labels=range(5))),
        "per_class_recall": recall_score(y_true, predicted, labels=range(5), average=None, zero_division=0).tolist(),
        "severe_cases_undergraded_below_moderate": int(np.sum(severity_mask & (predicted < 2))),
        "referable_dr_threshold": 0.5,
        "referable_dr_sensitivity": tp / (tp + fn) if tp + fn else None,
        "referable_dr_specificity": tn / (tn + fp) if tn + fp else None,
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=range(5)).tolist(),
        "classification_report": classification_report(y_true, predicted, labels=range(5), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }


def accuracy_interval(y_true, probabilities, seed=42, repeats=2000):
    """Image-level bootstrap interval; does not assume patient identifiers exist."""
    correct = np.asarray(y_true) == np.asarray(probabilities).argmax(axis=1)
    rng = np.random.default_rng(seed)
    samples = [float(correct[rng.integers(0, len(correct), len(correct))].mean()) for _ in range(repeats)]
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]
