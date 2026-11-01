"""Validation-only ablation. Never exports the broken model or reads test pixels."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from training.data_pipeline import load_pixels
from training.evaluation import metrics
from training.train import make_models, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("training/outputs/cpu_candidate"))
    args = parser.parse_args()
    os.environ.setdefault("KERAS_HOME", str(Path(".training-cache/keras").resolve()))
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    manifest = pd.read_csv(args.run_dir / "manifest.csv")
    train = manifest[manifest.split == "train"]
    validation = manifest[manifest.split == "validation"]
    _, extractor, head, _ = make_models(tf, corrected_normalization=False)
    train_features = extractor.predict(dataset(tf, load_pixels(train, 224)), verbose=1)
    val_features = extractor.predict(dataset(tf, load_pixels(validation, 224)), verbose=1)
    labels = train.label.to_numpy()
    weights = {i: float(np.sqrt(len(labels) / (5 * n))) for i, n in enumerate(np.bincount(labels, minlength=5))}
    head.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit(train_features, labels, batch_size=32, validation_data=(val_features, validation.label.to_numpy()),
             epochs=40, class_weight=weights, verbose=2,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6)])
    legacy = metrics(validation.label.to_numpy(), head.predict(val_features, verbose=0))
    corrected = json.loads((args.run_dir / "metrics.json").read_text())["frozen_validation"]
    report = {"comparison": "Matched frozen-backbone validation comparison of legacy vs corrected normalization recipe; includes normalization-state finalization and is not the historical trained artifact",
              "test_used": False, "same_manifest": True, "same_seed": 42,
              "old_normalization_validation": legacy, "corrected_normalization_validation": corrected,
              "accuracy_change_percentage_points": (corrected["accuracy"] - legacy["accuracy"]) * 100}
    (args.run_dir / "normalization_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
