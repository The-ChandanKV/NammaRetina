"""Train a production-compatible EfficientNet, with validation-only selection.

CPU: python -m training.train --data-root training/data/aptos224 --mode frozen
GPU: python -m training.train --data-root /path/to/aptos --mode finetune
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

import numpy as np

from training.data_pipeline import build_manifest, load_pixels, CLASS_NAMES
from training.evaluation import metrics, accuracy_interval


def uncompiled_classifier(tf, head):
    """Copy classifier weights without carrying nested optimizer/metric state."""
    clone = tf.keras.Sequential.from_config(head.get_config())
    clone.set_weights(head.get_weights())
    return clone


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=Path("APTOS-19"))
    parser.add_argument("--run-dir", type=Path, default=Path("training/outputs/candidate"))
    parser.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--fine-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Development run: do not inspect the held-out test set")
    return parser.parse_args()


def dataset(tf, pixels, labels=None, batch_size=32, shuffle=False, augment=False, seed=42):
    data = tf.data.Dataset.from_tensor_slices(pixels if labels is None else (pixels, labels))
    if shuffle:
        data = data.shuffle(len(pixels), seed=seed, reshuffle_each_iteration=True)
    data = data.batch(batch_size)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.08, fill_mode="constant", fill_value=128, seed=seed + 1),
        tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=128, seed=seed + 2),
        tf.keras.layers.RandomContrast(0.12, seed=seed + 3),
    ]) if augment else None

    def transform(x):
        x = tf.cast(x, tf.float32)
        if augmentation is not None:
            x = augmentation(x, training=True)
        return tf.clip_by_value(x, 0, 255) / 127.5 - 1.0

    if labels is None:
        data = data.map(transform, num_parallel_calls=2)
    else:
        data = data.map(lambda x, y: (transform(x), y), num_parallel_calls=2)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return data.with_options(options).prefetch(1)


def make_models(tf, corrected_normalization=True):
    inputs = tf.keras.Input((224, 224, 3), name="retina_input_minus_one_to_one")
    # Preserve the existing app API while giving ImageNet EfficientNet its
    # documented [0,255] input. EfficientNet performs its own normalization.
    restored = tf.keras.layers.Rescaling(127.5, offset=127.5, name="restore_efficientnet_pixel_range")(inputs) if corrected_normalization else inputs
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=restored)
    # Keras 3.3 H5 application-weight loading updates normalization variables
    # without refreshing cached broadcast tensors. Finalize now, just as .keras
    # restoration does, so features do not change after serialization.
    if corrected_normalization:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.Normalization):
                layer.finalize_state()
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="retina_features")(backbone.output)
    extractor = tf.keras.Model(inputs, pooled, name="retina_feature_extractor")
    extractor.trainable = False
    head = tf.keras.Sequential([
        tf.keras.Input((extractor.output_shape[-1],)),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(5, activation="softmax", dtype="float32"),
    ], name="dr_classifier")
    model = tf.keras.Model(inputs, head(extractor.output), name="nammaretina_efficientnetb0_v2")
    assert model.get_layer("top_conv") is not None
    return model, extractor, head, backbone


def main():
    args = arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "validation_history.jsonl").write_text("")
    os.environ.setdefault("KERAS_HOME", str(Path(".training-cache/keras").resolve()))
    import tensorflow as tf
    import keras

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    devices = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not devices:
        raise SystemExit("No GPU detected. Select a GPU runtime or omit --require-gpu for the CPU baseline.")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"TensorFlow {tf.__version__}; Keras {keras.__version__}; GPUs {devices}", flush=True)

    manifest, audit = build_manifest(args.data_root, args.split_root)
    manifest.to_csv(args.run_dir / "manifest.csv", index=False)
    (args.run_dir / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print("Data audit:", {k: v for k, v in audit.items() if k != "removed"}, flush=True)
    frames = {s: manifest[manifest.split == s].reset_index(drop=True) for s in ["train", "validation", "test"]}
    train_pixels = load_pixels(frames["train"], 224)
    val_pixels = load_pixels(frames["validation"], 224)
    train_labels = frames["train"].label.to_numpy()
    val_labels = frames["validation"].label.to_numpy()
    model, extractor, head, backbone = make_models(tf)

    started = time.monotonic()
    # Cache features, not augmented images: this makes a CPU baseline feasible.
    # The fingerprint includes pixel hashes, split labels and preprocessing code.
    fingerprint = hashlib.sha256((manifest.to_csv(index=False) + Path("retinal_preprocessing.py").read_text() + Path(__file__).read_text()).encode()).hexdigest()
    feature_file = args.run_dir / "frozen_features.npz"
    cached = np.load(feature_file) if feature_file.exists() else None
    if cached is not None and str(cached["fingerprint"]) == fingerprint:
        train_features, val_features = cached["train"], cached["validation"]
        cached.close()
        print("Reusing fingerprint-verified frozen features", flush=True)
    else:
        if cached is not None:
            cached.close()
        print("Extracting training features", flush=True)
        train_features = extractor.predict(dataset(tf, train_pixels, batch_size=args.batch_size), verbose=1)
        print("Extracting validation features", flush=True)
        val_features = extractor.predict(dataset(tf, val_pixels, batch_size=args.batch_size), verbose=1)
        np.savez_compressed(feature_file, train=train_features, validation=val_features, fingerprint=fingerprint)
    print(f"Feature extraction elapsed: {time.monotonic() - started:.1f}s", flush=True)

    class_counts = np.bincount(train_labels, minlength=5)
    # Square-root weighting is gentler than fully balancing this small dataset.
    class_weights = {i: float(np.sqrt(len(train_labels) / (5 * n))) for i, n in enumerate(class_counts)}
    head.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class ValidationMetrics(tf.keras.callbacks.Callback):
        def __init__(self, val_data, labels, stage):
            super().__init__()
            self.val_data, self.labels, self.stage = val_data, labels, stage

        def on_epoch_end(self, epoch, logs=None):
            values = self.model.predict(self.val_data, verbose=0)
            result = metrics(self.labels, values)
            if logs is not None:
                logs["val_qwk"] = result["quadratic_weighted_kappa"]
                logs["val_macro_f1"] = result["macro_f1"]
            with (args.run_dir / "validation_history.jsonl").open("a") as output:
                output.write(json.dumps({"stage": self.stage, "epoch": epoch + 1,
                                         "accuracy": result["accuracy"], "qwk": result["quadratic_weighted_kappa"],
                                         "macro_f1": result["macro_f1"]}) + "\n")
            print(f" validation QWK={result['quadratic_weighted_kappa']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)

    head_path = args.run_dir / "best_head.keras"
    head.fit(train_features, train_labels, batch_size=args.batch_size, validation_data=(val_features, val_labels),
             epochs=args.head_epochs, class_weight=class_weights, verbose=2,
             callbacks=[ValidationMetrics(val_features, val_labels, "frozen"),
                        tf.keras.callbacks.ModelCheckpoint(str(head_path), monitor="val_accuracy", mode="max", save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6)])
    best_head = tf.keras.models.load_model(head_path, compile=False)
    head.set_weights(best_head.get_weights())
    # The frozen head has its own compiled metrics. Do not embed these in a
    # separately compiled fine-tuning model: nested metric names can shadow
    # root metrics or fail when restored with unbuilt state.
    head = uncompiled_classifier(tf, head)
    model = tf.keras.Model(extractor.inputs, head(extractor.output), name="nammaretina_efficientnetb0_v2")
    baseline_val = metrics(val_labels, head.predict(val_features, verbose=0))
    model_path = args.run_dir / "efficientnetb0_dr.keras"
    model.save(model_path)
    # Check the exact release artifact before ever looking at test labels.
    probe_pixels = val_pixels[:8].astype(np.float32) / 127.5 - 1.0
    expected_predictions = model(probe_pixels, training=False).numpy()
    restored_model = tf.keras.models.load_model(model_path, compile=False)
    actual_predictions = restored_model(probe_pixels, training=False).numpy()
    roundtrip_error = float(np.max(np.abs(expected_predictions - actual_predictions)))
    if not np.allclose(expected_predictions, actual_predictions, atol=1e-5):
        raise RuntimeError(f"Predictions changed after serialization: {roundtrip_error}")
    selection = "frozen"

    if args.mode == "finetune":
        # Keep batch normalization frozen to avoid destroying pretrained stats.
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False
        for layer in backbone.layers[-60:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        val_ds = dataset(tf, val_pixels, val_labels, args.batch_size)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", mode="max", save_best_only=True)
        checkpoint.best = baseline_val["accuracy"]
        model.fit(dataset(tf, train_pixels, train_labels, args.batch_size, shuffle=True, augment=True, seed=args.seed),
                  validation_data=val_ds, epochs=args.fine_epochs, class_weight=class_weights, verbose=2,
                  callbacks=[ValidationMetrics(dataset(tf, val_pixels, batch_size=args.batch_size), val_labels, "finetune"),
                             checkpoint, tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=7),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-7)])
        selection = "best_validation_across_frozen_and_finetuned"

    best = tf.keras.models.load_model(model_path, compile=False)
    val_probs = best.predict(dataset(tf, val_pixels, batch_size=args.batch_size), verbose=1)
    report = {"validation": metrics(val_labels, val_probs), "frozen_validation": baseline_val,
              "frozen_serialization_max_probability_error": roundtrip_error,
              "selection_metric": "validation_accuracy", "selection": selection,
              "test_used_for_selection": False, "clinical_validation": False,
              "historical_notebook_accuracy": 0.7541,
              "historical_score_directly_comparable": False,
              "comparison_note": "No previous trained artifact is available; duplicate quarantine may change evaluation membership.",
              "training_seconds": time.monotonic() - started}
    if not args.no_test:
        print("Final held-out test evaluation (not used for model selection)", flush=True)
        test_pixels = load_pixels(frames["test"], 224)
        test_labels = frames["test"].label.to_numpy()
        test_probs = best.predict(dataset(tf, test_pixels, batch_size=args.batch_size), verbose=1)
        report["test"] = metrics(test_labels, test_probs)
        report["test"]["accuracy_bootstrap_95_interval"] = accuracy_interval(test_labels, test_probs, args.seed)
        predictions = frames["test"][["id_code", "label"]].copy()
        for i in range(5):
            predictions[f"probability_{i}"] = test_probs[:, i]
        predictions.to_csv(args.run_dir / "test_predictions.csv", index=False)

    # Verify that the serialized artifact can be loaded by the production loader
    # and that its top_conv graph supports a real Grad-CAM gradient.
    import model_loader
    import gradcam
    model_loader._model, model_loader._model_available = best, True
    probe = str(frames["validation"].iloc[0].path)
    inference = model_loader.predict_severity(probe)
    cam = gradcam._compute_gradcam(best, probe, inference["severity"], "top_conv")
    if not np.all(np.isfinite(cam)) or cam.max() <= 0:
        raise RuntimeError("Exported model failed the genuine Grad-CAM smoke check")
    report["production_smoke_test"] = {"reload": True, "prediction": inference["model_used"], "gradcam": True}
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {"model_file": model_path.name, "sha256": digest, "input_shape": [None, 224, 224, 3],
                "input_range": [-1, 1], "normalization_adapter": "restore_efficientnet_pixel_range",
                "preprocessing": "shared_ben_graham_v1", "class_names": CLASS_NAMES,
                "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
                "seed": args.seed, "mode": args.mode, "platform": platform.platform(),
                "manifest_sha256": hashlib.sha256((args.run_dir / "manifest.csv").read_bytes()).hexdigest(),
                "dataset_source": "https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data",
                "dataset_archive_sha256": "8a6c20b4b645efa026e1695511e5eadf8a8bf0dd5bbf5d906f35b53919020ce3",
                "test_evaluated": not args.no_test, "clinical_validation": False}
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (args.run_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.run_dir / "model.sha256").write_text(f"{digest}  {model_path.name}\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Exported {model_path}\nSHA256 {digest}", flush=True)


if __name__ == "__main__":
    main()
