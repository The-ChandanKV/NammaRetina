"""Regression coverage for standardized examination snapshots and real PDFs."""
import tempfile
import unittest
import re
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from reportlab.platypus import SimpleDocTemplate

from reporting import build_report_context, normalize_report, REPORT_SECTIONS, pdf_filename
from report_generator import generate_pdf


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.patient = {"patient_id": 5, "name": "A & B <Patient>", "age": None,
                        "gender": "Female", "diabetes_history": "Not provided"}
        self.report = {"report_id": 7, "patient_id": 5, "date": "2026-01-02",
                       "severity": 2, "confidence": .8, "llm_summary": "(pending)",
                       "recommendation": ""}

    def test_snapshot_excludes_future_visits_same_day_and_other_patients(self):
        rows = [dict(self.report, report_id=8, severity=4),
                dict(self.report, report_id=6, severity=3),
                dict(self.report, report_id=4, severity=1, date="2026-01-01"),
                dict(self.report, report_id=2, patient_id=9, severity=0),
                dict(self.report, report_id=3, severity=0, confidence=0)]
        context = build_report_context(self.report, self.patient, rows)
        self.assertEqual([r["report_id"] for r in context["longitudinal"]["examinations"]], [4, 6, 7])
        self.assertEqual(context["progression_status"], "Improved")
        self.assertEqual(context["longitudinal"]["previous_vs_current"]["previous"]["report_id"], 6)
        self.assertEqual(context["longitudinal"]["latest_exam"]["report_id"], 7)
        self.assertNotIn("4/4", " ".join(x["text"] for x in context["interpretation"]))
        self.assertEqual(context["llm_summary"], "")
        self.assertEqual(len(context["progression"]), 3)

    def test_invalid_predictions_have_no_normal_retina_explanation(self):
        for severity, confidence in [(None, .9), (8, .9), (2.5, .9), (0, None), (0, "nan"), (0, "bad"), (0, float("inf")), (0, -1)]:
            with self.subTest(severity=severity, confidence=confidence):
                context = build_report_context(dict(self.report, severity=severity, confidence=confidence), self.patient, [])
                self.assertFalse(context["report"]["model_used"])
                self.assertEqual(context["progression_status"], "Unavailable")
                self.assertEqual(context["report"]["severity_label"], "Analysis unavailable")
                self.assertIn("No valid model prediction", context["interpretation"][0]["text"])
                self.assertEqual(context["longitudinal"]["n_examinations"], 0)

    def test_legacy_percent_and_fraction_agree_without_mutation(self):
        original = dict(self.report, confidence="85.7", severity="2")
        a = normalize_report(original)
        b = normalize_report(dict(self.report, confidence=.857))
        self.assertEqual(a["confidence_pct"], b["confidence_pct"])
        self.assertEqual(a["confidence_pct"], 85.7)
        self.assertEqual(original["confidence"], "85.7")

    def test_real_pdf_all_sections_escape_text_wrap_and_ignore_corrupt_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.png"
            corrupt.write_bytes(b"not an image")
            image = Path(tmp) / "scan.png"
            Image.new("RGB", (320, 80), "red").save(image)
            context = build_report_context(dict(self.report, llm_summary="A <tag> & a > b\n" * 50),
                                           dict(self.patient, diabetes_history="Long clinical note <test> & value. " * 250), [])
            def uncompressed_doc(*args, **kwargs):
                return SimpleDocTemplate(*args, **kwargs, pageCompression=0)
            with patch("report_generator.SimpleDocTemplate", side_effect=uncompressed_doc):
                path = generate_pdf(7, self.patient, 2, "Moderate", .8, "", "",
                                    image_path=str(image), heatmap_path=str(corrupt),
                                    output_dir=tmp, report_context=context)
            data = Path(path).read_bytes()
            self.assertTrue(data.startswith(b"%PDF-"))
            self.assertTrue(data.rstrip().endswith(b"%%EOF"))
            self.assertEqual(Path(path).name, pdf_filename(7))
            for _, title in REPORT_SECTIONS:
                self.assertIn(title.encode(), data)
            self.assertIn(b"Image unavailable for this examination.", data)
            # PDF may split escaped XML entities into separate text runs.
            rendered_text = b"".join(re.findall(rb"\(((?:\\.|[^\\)])*)\)\s*Tj", data))
            self.assertTrue(b"A & B <Patient>" in rendered_text, "Patient text must remain literal")
            self.assertIn(b"/Subtype /Image", data)
            self.assertEqual(list(Path(tmp).glob(".report_*")), [])

    def test_failed_rebuild_keeps_previous_complete_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / pdf_filename(7)
            path.write_bytes(b"previous complete document")
            with patch("report_generator.SimpleDocTemplate.build", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    generate_pdf(7, self.patient, 2, "Moderate", .8, "", "", output_dir=tmp)
            self.assertEqual(path.read_bytes(), b"previous complete document")
            self.assertEqual(list(Path(tmp).glob(".report_*")), [])


if __name__ == "__main__":
    unittest.main()
