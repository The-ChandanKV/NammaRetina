"""Checks for issues found during the project-wide audit; no live services."""
import concurrent.futures
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

import database
import evaluation_reports
import explainability_evaluation
import gradcam
import lime_explainer
import llm_agent
import model_evaluation as evaluation
import model_loader
import simulation


class DatabaseAuditTests(unittest.TestCase):
    def test_public_list_helpers_and_adapter_fetch_all_pages(self):
        rows = [{"report_id": i, "patient_id": i} for i in range(1205)]
        class Query:
            def __init__(self): self.start = self.end = 0
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def order(self, *a, **k): return self
            def range(self, start, end): self.start, self.end = start, end; return self
            def execute(self): return SimpleNamespace(data=rows[self.start:self.end + 1])
        backend = SimpleNamespace(table=lambda _: Query())
        with patch.object(database, "_get_sb", return_value=backend):
            for get in [database.get_all_reports, database.get_all_patients_with_reports,
                        lambda: database.get_reports_for_patient(5), lambda: database.get_progression_history(5),
                        lambda: database._SupabasePseudoConn(backend).execute("SELECT * FROM Patients ORDER BY patient_id DESC").fetchall()]:
                self.assertEqual(len(get()), 1205)

    def test_storage_search_finds_objects_beyond_first_page(self):
        storage = Mock()
        storage.list.side_effect = [[{"name": "partial"}] * 100, [{"name": "target.png"}]]
        backend = Mock()
        backend.storage.from_.return_value = storage
        with patch.object(database, "_get_sb", return_value=backend):
            self.assertTrue(database.storage_object_exists("retinal-images/uploads/target.png"))
        self.assertEqual(storage.list.call_args.args[1]["offset"], 100)
        self.assertEqual(storage.list.call_args.args[1]["search"], "target.png")

    def test_unknown_user_count_fails_closed(self):
        backend = Mock()
        backend.table.return_value.select.return_value.execute.return_value.count = None
        with patch.object(database, "_get_sb", return_value=backend):
            with self.assertRaises(database.DatabaseUnavailable): database.count_users()


class EvaluationAuditTests(unittest.TestCase):
    def test_cluster_bootstrap_repeats_each_drawn_patient(self):
        # Deterministic draws: selecting patient A twice must repeat both A rows.
        rng = Mock()
        rng.integers.return_value = np.array([0, 0])
        seen = []
        def statistic(rows): seen.append(rows.tolist()); return float(np.mean(rows))
        with patch.object(np.random, "default_rng", return_value=rng):
            ci = evaluation.bootstrap_ci(np.array([1., 3., 9.]), statistic, n_boot=1, patient_ids=np.array(["A", "A", "B"]))
        self.assertEqual(seen, [[1., 3., 1., 3.]])
        self.assertEqual(ci, [2., 2.])

    def test_empty_and_noninteger_labels_are_rejected(self):
        for labels, probs in [([], np.empty((0, 5))), ([1.5], np.eye(5)[[1]]), ([[1]], np.eye(5)[[1]])]:
            with self.assertRaises(ValueError): evaluation.evaluate_predictions(labels, probs)
        with self.assertRaises(ValueError):
            evaluation.bootstrap_ci(np.ones(3), np.mean, n_boot=2, patient_ids=np.array(["A"]))

    def test_unavailable_binary_statistics_are_null_not_zero_or_nan(self):
        result = evaluation.evaluate_predictions([0, 0], np.eye(5)[[0, 0]])
        self.assertIsNone(result["referable_dr"]["sensitivity"])
        self.assertIsNone(result["referable_dr"]["sensitivity_ci95"])
        self.assertIsNone(result["five_class"]["quadratic_weighted_kappa"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(evaluation.binary_confusion(np.array([True]), np.array([False]))["f1"], 0)

    def test_run_names_cannot_escape_root_and_concurrent_runs_do_not_collide(self):
        result = evaluation.evaluate_predictions([0, 1, 2, 3, 4], np.eye(5), dataset_name="../../escape", split_name="../test")
        with tempfile.TemporaryDirectory() as tmp:
            a = evaluation.write_evaluation_run(result, tmp)
            b = evaluation.write_evaluation_run(result, tmp)
            self.assertNotEqual(a, b)
            self.assertEqual(a.parent, Path(tmp))
            self.assertTrue((a / "metrics.json").is_file())

    def test_legacy_test_split_and_missing_metrics_stay_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps({"test": {"accuracy": .8, "n": 10}}))
            result = evaluation.import_legacy_training_metrics(path)
        self.assertEqual(result["dataset"]["split"], "test")
        self.assertIsNone(result["five_class"]["macro_f1"])
        self.assertIsNone(result["five_class"]["balanced_accuracy"])

    def test_dashboard_skips_corrupt_newer_run(self):
        valid = json.dumps(evaluation.evaluate_predictions([0, 1, 2, 3, 4], np.eye(5)))
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in [("20260101_valid", valid), ("20260102_corrupt", '{'), ("20260103_incomplete", '{"status":"complete"}')]:
                root = Path(tmp) / name; root.mkdir(); (root / "metrics.json").write_text(content)
            self.assertEqual(evaluation_reports.latest_evaluation(tmp)["run_id"], "20260101_valid")

    def test_object_arrays_in_prediction_archives_are_not_unpickled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            np.savez(path, y_true=[0], probabilities=np.eye(5)[[0]], metadata=np.array({"anything": 1}, dtype=object))
            with self.assertRaises(ValueError): evaluation.load_predictions_npz(path)


class ExplanationAuditTests(unittest.TestCase):
    def test_gradcam_is_projected_into_crop_in_original_coordinates(self):
        original = np.zeros((160, 300, 3), np.uint8)
        original[30:130, 100:200] = 100
        scalar = np.ones((7, 7), np.float32)
        mapped = gradcam.project_attention(scalar, original)
        self.assertEqual(mapped.shape, (160, 300))
        self.assertEqual(float(mapped[:30].sum()), 0)
        self.assertEqual(float(mapped[:, :100].sum()), 0)
        self.assertTrue(np.all(mapped[30:130, 100:200] == 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.png"; cv2.imwrite(str(path), original)
            with patch.object(gradcam, "_compute_gradcam", return_value=scalar):
                overlay = gradcam.generate_gradcam(object(), str(path), 2, tmp)
            raw = cv2.imread(str(gradcam.attention_path_for_heatmap(overlay)), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.array_equal(raw, mapped * 255))
            self.assertEqual(cv2.imread(overlay).shape, original.shape)

    def test_simulation_rejects_rgb_overlays_and_empty_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "scan.png"; Image.new("RGB", (80, 80), "red").save(image)
            for mode, color in [("RGB", "red"), ("L", 0)]:
                heat = Path(tmp) / "heat.png"; Image.new(mode, (80, 80), color).save(heat)
                with self.assertRaises(ValueError): simulation.simulate_progression(str(image), str(heat), [{"status": "Initial"}])

    def test_fov_mask_follows_offcenter_fundus(self):
        image = np.zeros((160, 400, 3), np.uint8)
        cv2.circle(image, (75, 80), 60, (100, 100, 100), -1)
        fov = simulation._retinal_fov_mask(image)
        self.assertGreater(fov[80, 75], .9)
        self.assertEqual(fov[80, 300], 0)

    def test_constant_gradcam_diagnostic_and_deletion_auc(self):
        result = explainability_evaluation.gradcam_reproducibility(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertIsNone(result["pearson_r"])
        def predictor(batch): return np.tile([.2, .2, .2, .2, .2], (len(batch), 1))
        result = explainability_evaluation.deletion_auc(np.ones((4, 4, 3)), np.ones((4, 4)), predictor, 2, steps=2)
        self.assertAlmostEqual(result["deletion_auc"], .2)

    def test_fallback_explains_predictions_without_inventing_lesions(self):
        for grade in range(5):
            text = llm_agent._fallback_explanation(grade, .8, "Improved")
            self.assertIn("screening prediction", text)
            self.assertIn("grade decreased", text)
            for unsupported in ["micro-aneurysms", "haemorrhages", "blood vessel growth", "looks healthy"]:
                self.assertNotIn(unsupported, text)
        self.assertIn("No valid model prediction", llm_agent._fallback_explanation(None, 0, "Initial"))

    def test_corrupt_lime_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "lime_1.json").write_text("[]")
            Image.new("RGB", (8, 8)).save(Path(tmp) / "lime_1.png")
            self.assertIsNone(lime_explainer.load_lime_artifact(tmp, 1))


class ModelLoadingAuditTests(unittest.TestCase):
    def test_parallel_first_requests_load_only_once(self):
        with patch.object(model_loader, "_model_available", None), patch.object(model_loader, "_model", None):
            sentinel = object()
            initializations = []
            def load_once():
                if model_loader._model_available is None:
                    initializations.append(1)
                    time.sleep(.02)
                    model_loader._model, model_loader._model_available = sentinel, True
                return model_loader._model
            with patch.object(model_loader, "_load_model_once", side_effect=load_once) as load:
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    results = list(pool.map(lambda _: model_loader.load_model(), range(2)))
                self.assertTrue(all(value is sentinel for value in results))
                # Both callers enter, but only after the first has completed initialization.
                self.assertEqual(len(initializations), 1)

    def test_initial_failure_recovers_after_backoff(self):
        with patch.object(model_loader, "_model_available", False), patch.object(model_loader, "_last_failure", 10), patch.object(model_loader.time, "monotonic", return_value=45):
            def recover(): model_loader._model_available = True; return "loaded"
            with patch.object(model_loader, "_load_model_once", side_effect=recover):
                self.assertEqual(model_loader.load_model(), "loaded")
