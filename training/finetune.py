"""Resume the measured baseline for validation-selected convolutional fine-tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from training.data_pipeline import load_pixels
from training.evaluation import metrics, accuracy_interval
from training.train import dataset, uncompiled_classifier


def supplementary_images(root, manifest, limit=200):
    """Training-only EyePACS subset, with exact cross-source duplicate exclusion."""
    import cv2
    from retinal_preprocessing import prepare_retina
    blocked = set(manifest.pixel_sha256)
    prepared = set()
    for path in manifest.path:
        im = prepare_retina(cv2.imread(path))
        prepared.add(hashlib.sha256(im.tobytes()).hexdigest())
    mapping = {"No_DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferate_DR": 4, "Proliferative_DR": 4}
    counts = [0] * 5
    rows = []
    paths = sorted(Path(root).rglob("*.png"))
    rng = np.random.default_rng(42)
    for index in rng.permutation(len(paths)):
        path = paths[index]
        label = mapping.get(path.parent.name)
        if label is None or (limit and counts[label] >= limit):
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        raw = hashlib.sha256(image.tobytes()).hexdigest()
        cooked = hashlib.sha256(prepare_retina(image).tobytes()).hexdigest()
        if raw in blocked or cooked in prepared:
            continue
        blocked.add(raw)
        prepared.add(cooked)
        counts[label] += 1
        rows.append({"path": str(path.resolve()), "label": label, "source": "EyePACS-2015"})
    return pd.DataFrame(rows), counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("training/outputs/cpu_candidate"))
    p.add_argument("--run-dir", type=Path, default=Path("training/outputs/finetuned"))
    p.add_argument("--extra-root", type=Path)
    p.add_argument("--extra-per-class", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--train-last-layers", type=int, default=60)
    p.add_argument("--no-test", action="store_true")
    args = p.parse_args()
    os.environ.setdefault("KERAS_HOME", str(Path(".training-cache/keras").resolve()))
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.baseline / "manifest.csv")
    frame.to_csv(args.run_dir / "manifest.csv", index=False)
    train = frame[frame.split == "train"].copy()
    validation = frame[frame.split == "validation"]
    extra_counts = [0] * 5
    if args.extra_root:
        extra, extra_counts = supplementary_images(args.extra_root, frame, args.extra_per_class)
        if not len(extra):
            raise ValueError("No supplementary images found")
        extra.to_csv(args.run_dir / "supplementary_manifest.csv", index=False)
        train = pd.concat([train, extra], ignore_index=True)
    print("Training images", len(train), "supplementary counts", extra_counts, flush=True)
    pixels, val_pixels = load_pixels(train, 224), load_pixels(validation, 224)
    labels, val_labels = train.label.to_numpy(), validation.label.to_numpy()
    model = tf.keras.models.load_model(args.baseline / "efficientnetb0_dr.keras", compile=False)
    if model.name == "nammaretina_validation_selected_ensemble":
        raise ValueError("Fine-tune the single-network parents before ensemble selection, not the selected ensemble.")
    # Nested Sequential compile state can survive deserialization independently
    # of the root compile=False and expose an unbuilt metric to the root trainer.
    old_head = model.get_layer("dr_classifier")
    head = uncompiled_classifier(tf, old_head)
    model = tf.keras.Model(model.inputs, head(model.get_layer("retina_features").output), name="nammaretina_finetuned")
    model.trainable = True
    for i, layer in enumerate(model.layers):
        layer.trainable = layer.name == "dr_classifier" or (
            i >= len(model.layers) - args.train_last_layers and
            not isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Normalization, tf.keras.layers.Rescaling))
        )
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    val_ds = dataset(tf, val_pixels, val_labels, args.batch_size)
    initial = metrics(val_labels, model.predict(dataset(tf, val_pixels, batch_size=args.batch_size), verbose=0))
    artifact = args.run_dir / "efficientnetb0_dr.keras"
    model.save(artifact)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(artifact), monitor="val_accuracy", mode="max", save_best_only=True)
    checkpoint.best = initial["accuracy"]
    counts = np.bincount(labels, minlength=5)
    weights = {i: float((len(labels)/(5*n)) ** args.class_weight_power) for i,n in enumerate(counts)}
    print("Starting validation accuracy", initial["accuracy"], "trainable parameters", sum(int(np.prod(v.shape)) for v in model.trainable_weights), flush=True)
    started = time.monotonic()
    model.fit(dataset(tf, pixels, labels, args.batch_size, shuffle=True, augment=True), validation_data=val_ds,
              epochs=args.epochs, class_weight=weights, verbose=2,
              callbacks=[checkpoint, tf.keras.callbacks.CSVLogger(str(args.run_dir / "history.csv")),
                         tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, baseline=initial["accuracy"]),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)])
    best = tf.keras.models.load_model(artifact, compile=False)
    val_probs = best.predict(dataset(tf, val_pixels, batch_size=args.batch_size), verbose=0)
    report = {"validation": metrics(val_labels, val_probs), "initial_validation": initial,
              "supplementary_counts": extra_counts, "selection_metric": "validation_accuracy",
              "test_used_for_selection": False, "training_seconds": time.monotonic()-started,
              "target_accuracy": 0.90, "target_achieved_on_test": None}
    report["evaluation_note"] = "The APTOS benchmark was evaluated previously for the baseline; no test labels are used for gradient updates or checkpoint selection. Independent external evaluation is still required."
    report["evaluation_only"] = args.epochs == 0
    if args.epochs == 0 and (args.baseline / "metrics.json").exists():
        parent_report = json.loads((args.baseline / "metrics.json").read_text())
        report["supplementary_counts"] = parent_report.get("supplementary_counts", extra_counts)
        report["training_recipe"] = parent_report.get("recipe", parent_report.get("training_recipe"))
    report["recipe"] = {"learning_rate": args.learning_rate, "class_weight_power": args.class_weight_power,
                         "epochs_requested": args.epochs, "batch_size": args.batch_size,
                         "train_last_layers": args.train_last_layers}
    if not args.no_test:
        test = frame[frame.split == "test"]
        prob = best.predict(dataset(tf, load_pixels(test, 224), batch_size=args.batch_size), verbose=0)
        report["test"] = metrics(test.label.to_numpy(), prob)
        report["test"]["accuracy_bootstrap_95_interval"] = accuracy_interval(test.label.to_numpy(), prob)
        report["target_achieved_on_test"] = report["test"]["accuracy"] > 0.90
    import model_loader, gradcam
    model_loader._model, model_loader._model_available = best, True
    probe = str(validation.iloc[0].path)
    prediction = model_loader.predict_severity(probe)
    cam = gradcam._compute_gradcam(best, probe, prediction["severity"], "top_conv")
    if not np.isfinite(cam).all() or cam.max() <= 0:
        raise ValueError("Grad-CAM export check failed")
    report["production_smoke_test"] = {"reload": True, "prediction": True, "gradcam": True}
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    metadata = {"sha256": sha, "model_file": artifact.name, "input_shape": [None,224,224,3],
                "input_range": [-1,1], "tensorflow_version": tf.__version__, "class_order": [0,1,2,3,4],
                "preprocessing": "shared_ben_graham_v1", "clinical_validation": False,
                "parent_artifact_sha256": hashlib.sha256((args.baseline / "efficientnetb0_dr.keras").read_bytes()).hexdigest()}
    (args.run_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.run_dir / "model.sha256").write_text(sha + "  " + artifact.name + "\n")
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    frame.to_csv(args.run_dir / "manifest.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
