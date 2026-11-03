"""Offline regression checks. No Supabase or LLM requests are permitted."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["SKIP_BOOTSTRAP"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nammaretina-test-mpl")

import cv2
import numpy as np
from PIL import Image

import app as web
import database
import gradcam
import model_loader
import preprocess_local
import progression
import xai_engine
import httpx


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(database, "_get_sb", side_effect=AssertionError("Unexpected live DB call"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.tmp = tempfile.TemporaryDirectory(dir=web.BASE_DIR)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("UPLOAD_DIR", "HEATMAP_DIR", "REPORT_DIR", "SIMULATION_DIR"):
            target = self.root / name
            target.mkdir()
            self.patch(web, name, target)
        web.app.config.update(TESTING=True)
        self.client = web.app.test_client()
        self.patient = {"patient_id": 5, "name": "Test Patient", "age": 50,
                        "gender": "Female", "diabetes_history": "Test"}
        self.report = {"report_id": 7, "patient_id": 5, "severity": 2, "confidence": 0.8,
                       "date": "2026-01-01", "image_path": "retinal-images/uploads/scan.png",
                       "heatmap_path": "", "recommendation": "Consult specialist", "llm_summary": "Test summary"}

    def patch(self, target, name, value=None, **kwargs):
        p = patch.object(target, name, value, **kwargs) if value is not None else patch.object(target, name, **kwargs)
        result = p.start()
        self.addCleanup(p.stop)
        return result

    def login(self, role="doctor"):
        with self.client.session_transaction() as session:
            session.update(user_id="offline-user", role=role, csrf_token="test-csrf")

    def image(self):
        output = io.BytesIO()
        Image.new("RGB", (80, 80), (150, 60, 20)).save(output, format="PNG")
        output.seek(0)
        return output

    def form(self):
        return {"csrf_token": "test-csrf", "patient_name": "Test Patient", "patient_age": "50",
                "patient_gender": "Female", "retinal_image": (self.image(), "scan.png")}

    def connection(self):
        def execute(sql, params=()):
            if "FROM Reports WHERE report_id" in sql:
                return database._FakeResult([self.report])
            if "FROM Patients WHERE patient_id" in sql:
                return database._FakeResult([self.patient])
            if "FROM Progression" in sql:
                return database._FakeResult([])
            raise AssertionError(sql)
        return SimpleNamespace(execute=execute, close=lambda: None)

    def test_all_templates_compile(self):
        for name in web.app.jinja_env.list_templates():
            web.app.jinja_env.get_template(name)

    def test_private_static_files_are_blocked(self):
        for path in ["uploads/scan.png", "heatmaps/map.png", "reports/report_7.pdf", "simulations/scan.png", "css/../uploads/scan.png"]:
            self.assertEqual(self.client.get("/static/" + path).status_code, 404)
        with self.client.get("/static/css/app.css") as response:
            self.assertEqual(response.status_code, 200)

    def test_anonymous_routes_require_login(self):
        for path in ["/scan", "/dashboard", "/patients", "/reports", "/result/7", "/xai/7", "/download/7", "/files/uploads/scan.png"]:
            self.assertEqual(self.client.get(path).status_code, 302)

    def test_csrf_rejects_missing_and_invalid_token(self):
        self.login()
        for data in ({}, {"csrf_token": "wrong"}):
            self.assertEqual(self.client.post("/upload", data=data).status_code, 400)

    def test_login_form_contains_session_csrf(self):
        response = self.client.get("/login")
        with self.client.session_transaction() as session:
            token = session["csrf_token"]
        self.assertIn(token.encode(), response.data)

    def test_public_registration_blocked_after_bootstrap(self):
        self.patch(web, "count_users", return_value=1)
        self.assertEqual(self.client.get("/register").status_code, 302)

    def test_backend_outage_shows_service_message(self):
        self.patch(web, "count_users", side_effect=httpx.ConnectError("offline"))
        response = self.client.get("/register")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"Service temporarily unavailable", response.data)

    def test_health_reports_backend_and_model_outage(self):
        self.patch(web, "init_db", side_effect=httpx.ConnectError("offline"))
        self.patch(model_loader, "is_model_available", return_value=False)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json["database_available"])
        self.assertFalse(response.json["model_available"])

    def test_health_reports_ready_only_with_database_and_model(self):
        self.patch(web, "init_db", return_value="supabase")
        self.patch(model_loader, "is_model_available", return_value=True)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")

    def test_new_oauth_accounts_auto_provision_as_doctor(self):
        sb = Mock()
        empty = SimpleNamespace(data=[])
        sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = empty
        sb.table.return_value.select.return_value.ilike.return_value.limit.return_value.execute.return_value = empty
        sb.table.return_value.insert.return_value.execute.return_value = empty
        sb.auth.admin.list_users.return_value = SimpleNamespace(users=[])
        self.patch(database, "_get_sb", return_value=sb)
        self.patch(database, "count_users", return_value=1)
        result = database.get_or_create_sso_user("new-user", "new@example.com", full_name="New Doctor")
        self.assertEqual(result["role"], "doctor")
        self.assertEqual(result["user_id"], "new-user")
        sb.table.return_value.insert.assert_called_once()

    def test_oauth_links_preprovisioned_email(self):
        sb = Mock()
        empty = SimpleNamespace(data=[])
        role = SimpleNamespace(data=[{"username": "Friend", "role": "doctor"}])
        # 1) google uid has no role, 2) password uid has role, 3) google uid still no role before insert
        sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.side_effect = [
            empty, role, empty
        ]
        sb.table.return_value.insert.return_value.execute.return_value = empty
        sb.auth.admin.list_users.return_value = SimpleNamespace(
            users=[SimpleNamespace(id="pwd-uid", email="friend@gmail.com")]
        )
        self.patch(database, "_get_sb", return_value=sb)
        self.patch(database, "count_users", return_value=2)
        result = database.get_or_create_sso_user("google-uid", "friend@gmail.com")
        self.assertEqual(result["user_id"], "google-uid")
        self.assertEqual(result["username"], "Friend")
        self.assertEqual(result["role"], "doctor")
        sb.table.return_value.insert.assert_called_once()

    def test_oauth_retains_requested_destination(self):
        self.patch(web, "exchange_oauth_code", return_value=SimpleNamespace(id="u", email="u@example.com", user_metadata={}))
        self.patch(web, "get_or_create_sso_user", return_value={"user_id": "u", "username": "Doctor", "role": "doctor", "email": "u@example.com"})
        with self.client.session_transaction() as session:
            session["next_url"] = "/dashboard"
        self.assertEqual(self.client.get("/auth/callback?code=test").location, "/dashboard")

    def test_oauth_uses_separate_client(self):
        auth_client = Mock()
        self.patch(database, "_auth_client", return_value=auth_client)
        database.exchange_oauth_code("test-code", "test-verifier")
        auth_client.auth.exchange_code_for_session.assert_called_once_with({"auth_code": "test-code", "code_verifier": "test-verifier"})

    def test_administrator_keeps_session_when_creating_account(self):
        self.login("admin")
        self.patch(web, "count_users", return_value=1)
        self.patch(web, "create_user", return_value="new-doctor")
        response = self.client.post("/register", data={"csrf_token": "test-csrf", "username": "New Doctor", "email": "new@example.com", "password": "test-password", "confirm_password": "test-password"})
        self.assertEqual(response.location, "/dashboard")
        with self.client.session_transaction() as session:
            self.assertEqual(session["user_id"], "offline-user")
            self.assertEqual(session["role"], "admin")

    def test_logout_requires_csrf_protected_post(self):
        self.login()
        self.assertEqual(self.client.get("/logout").status_code, 405)
        self.assertEqual(self.client.post("/logout").status_code, 400)
        self.assertEqual(self.client.post("/logout", data={"csrf_token": "test-csrf"}).status_code, 302)
        with self.client.session_transaction() as session:
            self.assertNotIn("user_id", session)

    def test_missing_model_does_not_create_records(self):
        self.login()
        self.patch(web, "predict_severity", return_value={"model_used": False})
        create = self.patch(web, "add_patient")
        save = self.patch(web, "add_report")
        response = self.client.post("/upload", data=self.form())
        self.assertEqual(response.status_code, 302)
        create.assert_not_called()
        save.assert_not_called()
        self.assertEqual(list(web.UPLOAD_DIR.iterdir()), [])

    def test_invalid_demographics_do_not_run_model(self):
        self.login()
        predict = self.patch(web, "predict_severity")
        form = self.form()
        form["patient_age"] = "-4"
        self.client.post("/upload", data=form)
        predict.assert_not_called()

    def test_followup_preserves_id_and_actual_model_scores(self):
        self.login()
        self.patch(web, "get_patient", return_value=self.patient)
        create = self.patch(web, "add_patient")
        scores = [0.02, 0.03, 0.8, 0.1, 0.05]
        self.patch(web, "predict_severity", return_value={"model_used": True, "severity": 2, "confidence": 0.8, "severity_label": "Moderate", "probabilities": scores})
        self.patch(web, "load_model", return_value=object())
        self.patch(web, "generate_gradcam", return_value=None)
        upload = self.patch(web, "upload_file")
        save = self.patch(web, "add_report", return_value=7)
        track = self.patch(web, "track_progression", return_value={"status": "Initial"})
        self.patch(web, "plot_progression", return_value=None)
        self.patch(web, "get_progression_history", return_value=[])
        self.patch(web, "explain_report", return_value="Summary")
        self.patch(web, "update_report_llm_summary")
        self.patch(web, "generate_pdf", return_value=str(web.REPORT_DIR / "report_7.pdf"))
        form = self.form()
        form["patient_id"] = "5"
        response = self.client.post("/upload", data=form)
        self.assertEqual(response.location, "/result/7")
        create.assert_not_called()
        self.assertEqual(save.call_args.kwargs["patient_id"], 5)
        self.assertEqual(save.call_args.kwargs["heatmap_path"], "")
        self.assertEqual(json.loads((web.REPORT_DIR / "probabilities_7.json").read_text()), scores)
        track.assert_called_once_with(5, 2, visit_date=save.call_args.kwargs["date"], report_id=7)
        image_call = upload.call_args_list[0]
        self.assertEqual(Path(image_call.args[0]).name, Path(image_call.args[2]).name)

    def test_pdf_restored_after_restart(self):
        self.login()
        self.patch(web, "get_report", return_value=self.report)
        def download(storage, destination):
            self.assertEqual(storage, "reports/pdfs/report_7.pdf")
            Path(destination).write_bytes(b"%PDF-offline-test")
            return str(destination)
        self.patch(web, "download_storage_file", side_effect=download)
        response = self.client.get("/download/7")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"%PDF-offline-test")
        response.close()

    def test_storage_download_writes_complete_file(self):
        sb = Mock()
        sb.storage.from_.return_value.download.return_value = b"artifact"
        self.patch(database, "_get_sb", return_value=sb)
        destination = self.root / "scan.png"
        database.download_storage_file("retinal-images/uploads/scan.png", destination)
        self.assertEqual(destination.read_bytes(), b"artifact")
        sb.storage.from_.assert_called_once_with("retinal-images")

    def test_regenerated_pdf_is_uploaded(self):
        self.login()
        self.patch(web, "get_report", return_value=self.report)
        self.patch(web, "get_patient", return_value=self.patient)
        self.patch(web, "get_progression_history", return_value=[])
        self.patch(web, "ensure_simulation_for_report", return_value=None)
        self.patch(web, "simulation_filename_for_image", return_value=None)
        self.patch(web, "resolve_artifact", return_value=None)
        pdf = str(web.REPORT_DIR / "report_7.pdf")
        self.patch(web, "generate_pdf", return_value=pdf)
        upload = self.patch(web, "upload_file")
        response = self.client.post("/report/7/regenerate", data={"csrf_token": "test-csrf"})
        self.assertEqual(response.location, "/result/7")
        upload.assert_called_once_with(pdf, "reports", "pdfs/report_7.pdf")

    def test_missing_pdf_is_rebuilt_on_download(self):
        self.login()
        self.patch(web, "get_report", return_value=self.report)
        self.patch(web, "get_patient", return_value=self.patient)
        self.patch(web, "get_progression_history", return_value=[])
        self.patch(web, "ensure_simulation_for_report", return_value=None)
        self.patch(web, "simulation_filename_for_image", return_value=None)
        self.patch(web, "resolve_artifact", return_value=None)
        pdf = web.REPORT_DIR / "report_7.pdf"
        self.patch(web, "generate_pdf", side_effect=lambda **kwargs: (pdf.write_bytes(b"%PDF-rebuilt"), str(pdf))[1])
        upload = self.patch(web, "upload_file")
        response = self.client.get("/download/7")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"%PDF-rebuilt")
        upload.assert_called_once_with(str(pdf), "reports", "pdfs/report_7.pdf")
        response.close()

    def test_legacy_placeholder_pdf_and_xai_are_blocked(self):
        self.login()
        self.report["confidence"] = 0
        self.patch(web, "get_report", return_value=self.report)
        self.patch(web, "get_connection", return_value=self.connection())
        self.assertEqual(self.client.get("/download/7").location, "/result/7")
        self.assertEqual(self.client.get("/xai/7").location, "/result/7")

    def test_historical_result_uses_visit_status(self):
        self.login()
        self.patch(web, "get_connection", return_value=self.connection())
        self.patch(web, "get_reports_for_patient", return_value=[self.report, dict(self.report, report_id=8, severity=4, date="2026-02-01")])
        self.patch(web, "storage_path_to_signed_url", return_value="")
        response = self.client.get("/result/7")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"status-Initial", response.data)

    def test_reports_list_shows_all_reports(self):
        self.login()
        self.patch(web, "get_all_reports", return_value=[self.report])
        self.patch(web, "get_patient", return_value=self.patient)
        response = self.client.get("/reports")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Reports", response.data)
        self.assertIn(b"#7", response.data)
        self.assertIn(b"Test Patient", response.data)

    def test_patients_list_includes_status_summary(self):
        self.login()
        directory = [{
            "patient": self.patient,
            "reports": [self.report],
            "progression": [{"status": "Worsened"}],
            "latest_status": "Worsened",
            "latest_severity": 2,
            "latest_severity_label": "Moderate",
            "summary": "Moderate DR — worsening",
        }]
        self.patch(web, "_build_patient_directory", return_value=directory)
        response = self.client.get("/patients")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Patients", response.data)
        self.assertIn(b"Test Patient", response.data)
        self.assertIn(b"Moderate DR \xe2\x80\x94 worsening", response.data)

    def test_patient_status_summary_wording(self):
        self.assertEqual(web._patient_status_summary(None, "No scans", 0), "Awaiting first scan")
        self.assertEqual(web._patient_status_summary(2, "Worsened", 2), "Moderate DR — worsening")
        self.assertEqual(web._patient_status_summary(0, "Stable", 3), "No DR — stable")

    def test_followup_form_contains_patient_id(self):
        self.login()
        self.patch(web, "get_patient", return_value=self.patient)
        response = self.client.get("/scan?patient_id=5")
        self.assertIn(b'name="patient_id" value="5"', response.data)

    def test_public_landing_and_scan_auth(self):
        landing = self.client.get("/")
        self.assertEqual(landing.status_code, 200)
        self.assertIn(b"NammaRetina", landing.data)
        self.assertEqual(self.client.get("/scan").status_code, 302)
        self.login()
        scan = self.client.get("/scan")
        self.assertEqual(scan.status_code, 200)
        self.assertIn(b"uploadForm", scan.data)
        # Private artifacts must not be reachable via /landing/
        self.assertEqual(self.client.get("/landing/../uploads/scan.png").status_code, 404)
        self.assertEqual(self.client.get("/static/landing/index.html").status_code, 404)

    def test_descending_progression_and_ties(self):
        query = Mock()
        for method in ("select", "eq", "order"):
            getattr(query, method).return_value = query
        query.execute.return_value.data = []
        sb = SimpleNamespace(table=lambda _: query)
        database._SupabasePseudoConn(sb).execute("SELECT * FROM Progression WHERE patient_id = ? ORDER BY date DESC", (5,))
        self.assertEqual(query.order.call_args_list[0].kwargs, {"desc": True})
        self.assertEqual(query.order.call_args_list[1].args, ("progression_id",))

    def test_temporary_sqlite_path_never_calls_supabase(self):
        for call in [lambda: database.init_db(db_path="test.db"), lambda: database.get_connection("test.db"),
                     lambda: database.add_patient("Test", 50, "Female", "Test", db_path="test.db"),
                     lambda: database.get_reports_for_patient(5, db_path="test.db")]:
            with self.assertRaises(ValueError):
                call()

    def test_storage_failure_propagates(self):
        path = self.root / "scan.png"
        path.write_bytes(b"test")
        sb = Mock()
        sb.storage.from_.return_value.upload.side_effect = RuntimeError("Storage unavailable")
        self.patch(database, "_get_sb", return_value=sb)
        with self.assertRaises(RuntimeError):
            database.upload_file(str(path), "retinal-images", "uploads/scan.png")

    def test_training_and_inference_pixels_match(self):
        path = self.root / "scan.png"
        image = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(image, (80, 60), 55, (60, 90, 170), -1)
        cv2.imwrite(str(path), image)
        offline, reason = preprocess_local.preprocess_single(str(path))
        expected = cv2.cvtColor(offline, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        np.testing.assert_array_equal(model_loader.preprocess_image(str(path))[0], expected)
        self.assertEqual(reason, "ok")

    def test_gradcam_failure_is_unavailable(self):
        path = self.root / "scan.png"
        Image.new("RGB", (80, 80), "red").save(path)
        self.patch(gradcam, "_compute_gradcam", side_effect=RuntimeError("Disconnected graph"))
        self.assertIsNone(gradcam.generate_gradcam(object(), str(path), 2, str(self.root)))
        self.assertFalse((self.root / "scan_gradcam.png").exists())

    def test_xai_does_not_invent_probabilities(self):
        self.patch(xai_engine, "compute_superpixel_saliency", return_value={})
        self.patch(xai_engine, "detect_lesions", return_value={})
        self.patch(xai_engine, "analyse_channel_features", return_value={})
        result = xai_engine.explain_prediction("unused", None, 2, 0.8)
        self.assertEqual(result["prob_chart"], [])
        self.assertEqual(result["decomposition"], [])
        self.assertEqual(result["counterfactuals"], [])
        scores = [0.02, 0.03, 0.8, 0.1, 0.05]
        result = xai_engine.explain_prediction("unused", None, 2, 0.8, scores)
        self.assertEqual([item["prob"] for item in result["prob_chart"]], [2, 3, 80, 10, 5])

    def test_progression_compares_explicit_prior_report(self):
        self.patch(progression, "get_patient", return_value=self.patient)
        self.patch(progression, "get_reports_for_patient", return_value=[
            dict(self.report, report_id=6, severity=3, date="2025-12-01"),
            self.report,
            dict(self.report, report_id=8, severity=4, date="2026-02-01"),
        ])
        self.patch(progression, "add_progression_record")
        result = progression.track_progression(5, 2, "2026-01-01", report_id=7)
        self.assertEqual(result["status"], "Improved")
        self.assertEqual(result["previous_severity"], 3)


if __name__ == "__main__":
    unittest.main()
