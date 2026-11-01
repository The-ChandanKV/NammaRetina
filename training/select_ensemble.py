"""Choose a small convex ensemble on validation data, then assess test once."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.data_pipeline import load_pixels
from training.evaluation import metrics, accuracy_interval
from training.train import dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("training/outputs/cpu_candidate"))
    p.add_argument("--candidate", type=Path, default=Path("training/outputs/eyepacs_finetuned"))
    p.add_argument("--run-dir", type=Path, default=Path("training/outputs/selected_candidate"))
    p.add_argument("--no-test", action="store_true")
    args = p.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.baseline / "manifest.csv")
    val = frame[frame.split == "validation"]
    x = load_pixels(val, 224)
    a = tf.keras.models.load_model(args.baseline / "efficientnetb0_dr.keras", compile=False)
    b = tf.keras.models.load_model(args.candidate / "efficientnetb0_dr.keras", compile=False)
    pa = a.predict(dataset(tf, x), verbose=0)
    pb = b.predict(dataset(tf, x), verbose=0)
    trials = []
    # Prefer a single network when validation accuracy is tied.
    for alpha in [1.0, 0.0, 0.25, 0.5, 0.75]:
        result = metrics(val.label.to_numpy(), alpha * pa + (1-alpha) * pb)
        trials.append({"baseline_weight": alpha, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    chosen = max(trials, key=lambda t: t["accuracy"])
    alpha = chosen["baseline_weight"]
    if alpha == 1:
        selected = a
    elif alpha == 0:
        selected = b
    else:
        # Keep the baseline's convolutional graph at the top level for Grad-CAM.
        # Grad-CAM targets this branch; it is not an attribution of both branches.
        other = b(a.input)
        output = tf.keras.layers.Add(name="ensemble_probabilities")([
            tf.keras.layers.Rescaling(alpha, name="baseline_probability_weight")(a.output),
            tf.keras.layers.Rescaling(1-alpha, name="candidate_probability_weight")(other),
        ])
        selected = tf.keras.Model(a.inputs, output, name="nammaretina_validation_selected_ensemble")
    artifact = args.run_dir / "efficientnetb0_dr.keras"
    selected.save(artifact)
    probe = x[:4].astype(np.float32)/127.5-1
    expected = selected(probe, training=False).numpy()
    restored = tf.keras.models.load_model(artifact, compile=False)
    actual = restored(probe, training=False).numpy()
    np.testing.assert_allclose(expected, actual, atol=1e-5)
    report = {"validation_trials": trials, "baseline_weight": alpha,
              "selection_metric": "validation_accuracy", "test_used_for_selection": False,
              "validation": metrics(val.label.to_numpy(), alpha*pa+(1-alpha)*pb),
              "target_accuracy": 0.9, "target_achieved_on_test": None,
              "clinical_validation": False,
              "gradcam_scope": "baseline_branch" if 0 < alpha < 1 else "single_network"}
    report["parent_artifact_sha256"] = {
        "baseline": hashlib.sha256((args.baseline/"efficientnetb0_dr.keras").read_bytes()).hexdigest(),
        "candidate": hashlib.sha256((args.candidate/"efficientnetb0_dr.keras").read_bytes()).hexdigest()}
    if (args.candidate/"metrics.json").exists():
        parent = json.loads((args.candidate/"metrics.json").read_text())
        report["candidate_training_recipe"] = parent.get("recipe")
        report["supplementary_counts"] = parent.get("supplementary_counts")
    report["evaluation_note"] = "This APTOS benchmark was previously evaluated for the baseline. Test labels are not used for fit or checkpoint/ensemble selection; independent external confirmation is still needed."
    import gradcam
    from model_loader import preprocess_image
    path = str(val.iloc[0].path)
    label = int(np.argmax(restored(preprocess_image(path),training=False).numpy()[0]))
    cam = gradcam._compute_gradcam(restored,path,label,"top_conv")
    if not np.isfinite(cam).all() or cam.max() <= 0:
        raise ValueError("Grad-CAM check failed")
    report["production_smoke_test"] = {"reload":True,"prediction":True,"gradcam":True}
    if not args.no_test:
        test = frame[frame.split == "test"]
        probs = restored.predict(dataset(tf, load_pixels(test,224)), verbose=0)
        report["test"] = metrics(test.label.to_numpy(), probs)
        report["test"]["accuracy_bootstrap_95_interval"] = accuracy_interval(test.label.to_numpy(), probs)
        report["target_achieved_on_test"] = report["test"]["accuracy"] > 0.9
        result = test[["id_code","label"]].copy()
        for i in range(5): result[f"probability_{i}"] = probs[:,i]
        result.to_csv(args.run_dir/"test_predictions.csv",index=False)
    report["model_bytes"] = artifact.stat().st_size
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (args.run_dir/"model.sha256").write_text(sha+"  "+artifact.name+"\n")
    (args.run_dir/"metrics.json").write_text(json.dumps(report,indent=2))
    (args.run_dir/"model_metadata.json").write_text(json.dumps({"sha256":sha,"input_shape":[None,224,224,3],"input_range":[-1,1],"gradcam_scope":report["gradcam_scope"],"clinical_validation":False},indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__": main()
