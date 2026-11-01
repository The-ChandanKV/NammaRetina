"""Data and evaluation checks that do not require TensorFlow or network access."""

import tempfile
import unittest
from pathlib import Path
import zipfile
import io
import hashlib
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from training.data_pipeline import build_manifest, SPLITS
from training.download_data import extract_checked
from training.evaluation import metrics, accuracy_interval
import model_loader
from training.finetune import supplementary_images


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        rng = np.random.default_rng(42)
        for split, csv in SPLITS.items():
            rows = []
            for label in range(5):
                for sample in range(2):
                    name = f"{split}_{label}_{sample}"
                    cv2.imwrite(str(self.images / f"{name}.png"), rng.integers(0, 255, (60, 60, 3), dtype=np.uint8))
                    rows.append({"id_code": name, "diagnosis": label})
            pd.DataFrame(rows).to_csv(self.root / csv, index=False)

    def test_test_duplicate_never_leaks_into_training(self):
        original = self.images / "test_0_0.png"
        (self.images / "train_0_0.png").write_bytes(original.read_bytes())
        manifest, audit = build_manifest(self.images, self.root)
        self.assertIn("test_0_0", set(manifest.id_code))
        self.assertNotIn("train_0_0", set(manifest.id_code))
        self.assertEqual(audit["removed_images"], 1)
        self.assertEqual(manifest.group_id.nunique(), len(manifest))

    def test_supplementary_images_exclude_evaluation_duplicates(self):
        manifest, _ = build_manifest(self.images, self.root)
        extra = self.root / "eyepacs" / "Mild"
        extra.mkdir(parents=True)
        (extra / "duplicate.png").write_bytes((self.images / "test_1_0.png").read_bytes())
        rng = np.random.default_rng(101)
        cv2.imwrite(str(extra / "new_patient_left.png"), rng.integers(0, 255, (60, 60, 3), dtype=np.uint8))
        retained, counts = supplementary_images(extra.parent, manifest)
        self.assertEqual(counts, [0, 1, 0, 0, 0])
        self.assertEqual(Path(retained.iloc[0].path).stem, "new_patient_left")

    def test_conflicting_duplicate_labels_are_quarantined(self):
        (self.images / "train_0_0.png").write_bytes((self.images / "validation_1_0.png").read_bytes())
        manifest, audit = build_manifest(self.images, self.root)
        self.assertNotIn("train_0_0", set(manifest.id_code))
        self.assertNotIn("validation_1_0", set(manifest.id_code))
        self.assertEqual(audit["conflicting_label_images"], 2)

    def test_split_id_overlap_is_rejected(self):
        path = self.root / SPLITS["test"]
        frame = pd.read_csv(path)
        frame.loc[0, "id_code"] = "train_0_0"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_manifest(self.images, self.root)

    def test_archive_traversal_is_rejected(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../outside.txt", "unsafe")
        with self.assertRaises(ValueError):
            extract_checked(archive, self.root / "destination")

    def test_perfect_model_has_perfect_metrics(self):
        truth = np.arange(5)
        probabilities = np.eye(5)
        result = metrics(truth, probabilities)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["quadratic_weighted_kappa"], 1)
        self.assertEqual(result["referable_dr_sensitivity"], 1)
        self.assertEqual(accuracy_interval(truth, probabilities, repeats=20), [1, 1])

    def test_invalid_model_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics([0], [[0.2] * 4])
        with self.assertRaises(ValueError):
            metrics([0], [[0.5] * 5])

    def test_wrong_model_download_never_replaces_existing_artifact(self):
        model = self.root / "model.keras"
        model.write_bytes(b"previous artifact")
        with patch.object(model_loader, "MODEL_SHA256", "a" * 64), patch.object(model_loader, "MODEL_URL", "https://example.com/model.keras"), patch("urllib.request.urlopen", return_value=io.BytesIO(b"HTML error page")):
            with self.assertRaises(ValueError):
                model_loader.ensure_model_file(model)
        self.assertEqual(model.read_bytes(), b"previous artifact")
        self.assertEqual(list(self.root.glob("model-download-*")), [])

    def test_verified_download_replaces_artifact_atomically(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("config.json", "{}")
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        model = self.root / "model.keras"
        with patch.object(model_loader, "MODEL_SHA256", digest), patch.object(model_loader, "MODEL_URL", "https://example.com/model.keras"), patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
            self.assertEqual(model_loader.ensure_model_file(model), model)
        self.assertEqual(model_loader.file_sha256(model), digest)

    def test_direct_url_with_id_is_not_mistaken_for_google_drive(self):
        self.assertIsNone(model_loader._drive_file_id("https://example.com/model?id=123"))
        self.assertEqual(model_loader._drive_file_id("https://drive.google.com/file/d/123/view"), "123")


if __name__ == "__main__":
    unittest.main()
