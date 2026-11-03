"""Tests for longitudinal analysis, genuine LIME, and clinical evaluation metrics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

os.environ["SKIP_BOOTSTRAP"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nammaretina-test-mpl")

import app as web
import explainability_evaluation
import lime_explainer
import model_evaluation
import progression_analysis
import simulation


class ProgressionAnalysisTests(unittest.TestCase):
    def test_longitudinal_deltas_and_intervals(self):
        reports = [
            {"report_id": 1, "severity": 1, "confidence": 0.9, "date": "2024-01-01", "image_path": "a"},
            {"report_id": 2, "severity": 2, "confidence": 0.8, "date": "2024-04-01", "image_path": "b"},
            {"report_id": 3, "severity": 2, "confidence": 0.7, "date": "2024-07-01", "image_path": "c"},
        ]
        result = progression_analysis.analyse_longitudinal_progression(
            reports, patient={"diabetes_history": "Type 2, 8 years"}
        )
        self.assertEqual(result["n_examinations"], 3)
        self.assertEqual(len(result["transitions"]), 2)
        self.assertEqual(result["transitions"][0]["ordinal_delta"], 1)
        self.assertEqual(result["transitions"][0]["interval_days"], 91)
        self.assertEqual(result["transitions"][1]["change"], "Unchanged")
        self.assertFalse(result["forecasting"]["available"])
        self.assertEqual(len(result["risk_context"]["available_factors"]), 1)
        self.assertIn("diabetes_duration", result["risk_context"]["unavailable_structured_fields"])

    def test_does_not_invent_risk_percentages(self):
        result = progression_analysis.analyse_longitudinal_progression(
            [{"report_id": 1, "severity": 0, "confidence": 0.9, "date": "2024-01-01"}],
            patient={"diabetes_history": "Not provided"},
        )
        self.assertIsNone(result["risk_context"]["scoring_method"])
        self.assertEqual(result["risk_context"]["available_factors"], [])


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.image = root / "scan.png"
        self.heat = root / "heat.png"
        Image.new("RGB", (64, 64), (180, 40, 30)).save(self.image)
        Image.new("L", (64, 64), 200).save(self.heat)
        self.patcher = patch.object(simulation.config, "SIMULATION_DIR", str(root / "sims"))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_deterministic_and_labeled(self):
        history = [{"status": "Worsened", "current_severity": 2}]
        path1 = simulation.simulate_progression(str(self.image), str(self.heat), history)
        path2 = simulation.simulate_progression(str(self.image), str(self.heat), history)
        self.assertEqual(Path(path1).read_bytes(), Path(path2).read_bytes())
        img = Image.open(path1)
        # Caption bar occupies the bottom strip.
        self.assertTrue(Path(path1).exists())
        self.assertIn("Not a Clinical Forecast", simulation.ILLUSTRATIVE_CAPTION)


class LimeExplainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "retina.png"
        # Synthetic fundus-like disk
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:224, :224]
        mask = (yy - 112) ** 2 + (xx - 112) ** 2 <= 100 ** 2
        arr[mask] = (40, 120, 80)
        arr[80:120, 90:140] = (200, 60, 60)
        Image.fromarray(arr).save(self.image)

        def predict_fn(batch: np.ndarray) -> np.ndarray:
            # Higher score when warm (reddish) pixels remain visible.
            rgb = ((batch + 1.0) * 127.5).clip(0, 255)
            red = rgb[..., 0].mean(axis=(1, 2))
            p = np.clip(red / 255.0, 0.05, 0.95)
            out = np.zeros((len(batch), 5), dtype=np.float64)
            out[:, 2] = p
            out[:, 0] = 1.0 - p
            return out

        self.predict_fn = predict_fn

    def test_lime_is_perturbation_based_and_reproducible(self):
        a = lime_explainer.explain_image_lime(
            str(self.image), self.predict_fn, target_class=2, n_samples=60, n_segments=25, seed=7, batch_size=10
        )
        b = lime_explainer.explain_image_lime(
            str(self.image), self.predict_fn, target_class=2, n_samples=60, n_segments=25, seed=7, batch_size=10
        )
        self.assertEqual(a["method"], "lime_superpixel_ridge")
        self.assertEqual(a["target_class"], 2)
        self.assertEqual(a["coefficients"], b["coefficients"])
        self.assertEqual(a["overlay_rgb"].shape, (224, 224, 3))
        self.assertTrue(np.isfinite(a["fidelity_r2"]))
        self.assertGreaterEqual(len(a["top_regions"]), 1)

    def test_lime_cache_roundtrip(self):
        result = lime_explainer.explain_image_lime(
            str(self.image), self.predict_fn, target_class=2, n_samples=40, n_segments=16, seed=1
        )
        saved = lime_explainer.save_lime_artifact(result, self.tmp.name, report_id=99)
        loaded = lime_explainer.load_lime_artifact(self.tmp.name, 99)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["target_class"], 2)
        self.assertTrue(Path(saved["overlay_path"]).exists())

    def test_stability_diagnostic_runs(self):
        stats = explainability_evaluation.lime_stability(
            str(self.image), self.predict_fn, target_class=2, seeds=(0, 1), n_samples=40, n_segments=16
        )
        self.assertIn("mean_top_jaccard", stats)


class EvaluationMetricsTests(unittest.TestCase):
    def test_five_class_and_binary_metrics(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0, 0, 1, 2, 2, 3, 4, 4, 1, 2])
        logits = rng.normal(size=(len(y_true), 5))
        # Bias toward true class for a solvable case.
        for i, y in enumerate(y_true):
            logits[i, y] += 2.5
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        result = model_evaluation.evaluate_predictions(
            y_true, probs, dataset_name="fixture", split_name="unit", seed=0
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["five_class"]["n"], 10)
        self.assertEqual(len(result["five_class"]["confusion_matrix"]), 5)
        self.assertIn("sensitivity", result["referable_dr"])
        self.assertIn("sensitivity_ci95", result["referable_dr"])
        self.assertEqual(result["referable_dr"]["definition"].startswith("Referable"), True)
        self.assertIn("expected_calibration_error", result["calibration"])
        self.assertEqual(result["external_validation"]["status"], "pending")

    def test_write_run_is_timestamped(self):
        y_true = np.array([0, 1, 2, 3, 4])
        probs = np.eye(5)
        result = model_evaluation.evaluate_predictions(y_true, probs, dataset_name="unit", split_name="t")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = model_evaluation.write_evaluation_run(result, tmp)
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "report.md").exists())
            payload = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(payload["dataset"]["n"], 5)


class FlaskIntegrationSmokeTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("database._get_sb", side_effect=AssertionError("Unexpected live DB call"))
        self.network.start()
        self.addCleanup(self.network.stop)
        web.app.config.update(TESTING=True)
        self.client = web.app.test_client()

    def test_evaluation_requires_admin(self):
        with self.client.session_transaction() as session:
            session.update(user_id="doc", role="doctor", csrf_token="x", username="Doc")
        response = self.client.get("/evaluation")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.location)

    def test_evaluation_admin_ok_without_fabricated_metrics(self):
        with self.client.session_transaction() as session:
            session.update(user_id="admin", role="admin", csrf_token="x", username="Admin")
        with patch("app.ensure_seed_evaluation", return_value=None), patch("app.latest_evaluation", return_value=None), patch("app.list_evaluation_runs", return_value=[]):
            response = self.client.get("/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No validated evaluation results available", response.data)


if __name__ == "__main__":
    unittest.main()
