"""
NammaRetina - Flask Web Application (Phases 10-11)
Central web application wiring together all pipeline components:
    Upload -> Preprocess -> Model -> Grad-CAM -> Progression -> Simulation
    -> Treatment -> LLM -> DB -> PDF Report
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename
from PIL import Image

import config
from database import (
    add_patient,
    add_report,
    count_users,
    create_user,
    get_connection,
    get_patient,
    get_progression_history,
    get_reports_for_patient,
    init_db,
    verify_user,
    upload_file,
    get_signed_url,
    storage_path_to_signed_url,
    update_report_llm_summary,
    BUCKET_IMAGES,
    BUCKET_HEATMAPS,
    BUCKET_REPORTS,
    get_oauth_sign_in_url,
    exchange_oauth_code,
    get_user_from_access_token,
    get_or_create_sso_user,
)
from gradcam import generate_gradcam
from llm_agent import explain_report
from model_loader import load_model, predict_severity
from progression import plot_progression, track_progression
from report_generator import generate_pdf
from simulation import simulate_progression
from treatment import get_treatment_recommendation
from xai_engine import explain_prediction

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=config.TEMPLATE_DIR,
    static_folder=config.STATIC_DIR,
)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["TEMPLATES_AUTO_RELOAD"] = True  # always read templates fresh from disk

if config.SECRET_KEY_IS_DEFAULT:
    logger.warning(
        "SECRET_KEY is the built-in development default. Set the SECRET_KEY "
        "environment variable to a random secret before deploying — sessions "
        "signed with the default key are trivially forgeable."
    )

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}

# Ensure directories exist
BASE_DIR = Path(config.BASE_DIR)
UPLOAD_DIR = Path(config.UPLOAD_DIR)
HEATMAP_DIR = Path(config.HEATMAP_DIR)
REPORT_DIR = Path(config.REPORT_DIR)
SIMULATION_DIR = Path(config.SIMULATION_DIR)

for d in [UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR, SIMULATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def db_path_to_url(rel_path: str | None) -> str:
    """Convert a DB-stored relative path like 'static\\uploads\\file.png' into a URL."""
    if not rel_path:
        return url_for("serve_file", subdir="placeholder", filename="none.svg")
    normalised = str(rel_path).replace("\\", "/")
    if normalised.startswith("static/"):
        normalised = normalised[len("static/"):]
    parts = normalised.split("/")
    if len(parts) >= 2:
        subdir = parts[-2]
        filename = parts[-1]
    else:
        subdir = "uploads"
        filename = parts[-1]
    return url_for("serve_file", subdir=subdir, filename=filename)


@app.template_filter("file_url")
def file_url_filter(rel_path: str | None) -> str:
    return db_path_to_url(rel_path)


# ─── Routes ───────────────────────────────────────────────────────────────────


def login_required(view):
    """Redirect anonymous users to /login, preserving the intended destination."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    """Create a new account. The first account created becomes the admin."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("Username, email and password are required.", "danger")
            return redirect(url_for("register"))
        if "@" not in email:
            flash("Please enter a valid email address.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        role = "admin" if count_users() == 0 else "doctor"
        try:
            user_id = create_user(username, password, email=email, role=role)
        except Exception as exc:
            flash(f"Registration failed: {exc}", "danger")
            return redirect(url_for("register"))

        session.clear()
        session["user_id"] = user_id
        session["username"] = username
        session["role"] = role
        flash(f"Welcome, {username}! Your account has been created.", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate via Supabase Auth (email + password)."""
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = verify_user(email, password)
        if user is None:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        session.clear()
        session["user_id"] = user["user_id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        next_url = request.args.get("next") or ""
        if not next_url.startswith("/") or next_url.startswith("//"):
            next_url = url_for("index")
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(next_url)

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Clear the session and return to the login page."""
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/auth/google")
def auth_google():
    """Initiate Google OAuth SSO flow via Supabase."""
    next_url = request.args.get("next") or ""
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        session["next_url"] = next_url

    callback_url = url_for("auth_callback", _external=True)
    try:
        oauth_url, verifier = get_oauth_sign_in_url(redirect_to=callback_url, provider="google")
        session["oauth_code_verifier"] = verifier
        return redirect(oauth_url)
    except Exception as exc:
        logger.error("Failed to generate Google OAuth URL: %s", exc)
        flash(f"Could not initiate Google sign in: {exc}", "danger")
        return redirect(url_for("login"))


@app.route("/auth/callback")
def auth_callback():
    """OAuth callback endpoint: completes SSO flow and provisions session."""
    error = request.args.get("error_description") or request.args.get("error")
    if error:
        logger.warning("OAuth error returned: %s", error)
        if "not enabled" in error.lower() or "unsupported provider" in error.lower():
            flash("Google Sign-In is not enabled yet in your Supabase project. In Supabase Dashboard, go to Authentication → Providers → Google and enable it.", "warning")
        else:
            flash(f"Google sign in failed: {error}", "danger")
        return redirect(url_for("login"))

    code = request.args.get("code")
    access_token = request.args.get("access_token")

    # If neither query param is present, render helper page to parse client hash fragments (#access_token=...)
    if not code and not access_token:
        return render_template("auth_callback.html")

    user = None
    if code:
        verifier = session.pop("oauth_code_verifier", None)
        try:
            user = exchange_oauth_code(code=code, verifier=verifier)
        except Exception as exc:
            logger.error("Failed to exchange OAuth code: %s", exc)
            flash(f"Authentication failed: {exc}", "danger")
            return redirect(url_for("login"))
    elif access_token:
        try:
            user = get_user_from_access_token(access_token)
        except Exception as exc:
            logger.error("Failed to retrieve user from access token: %s", exc)
            flash(f"Authentication failed: {exc}", "danger")
            return redirect(url_for("login"))

    if not user:
        flash("Could not retrieve your user details from Google.", "danger")
        return redirect(url_for("login"))

    full_name = None
    if hasattr(user, "user_metadata") and isinstance(user.user_metadata, dict):
        full_name = user.user_metadata.get("full_name") or user.user_metadata.get("name")

    try:
        user_record = get_or_create_sso_user(
            user_id=str(user.id),
            email=str(user.email) if user.email else "",
            full_name=full_name,
        )
    except Exception as exc:
        logger.error("Failed to provision SSO user: %s", exc)
        flash(f"Account setup failed: {exc}", "danger")
        return redirect(url_for("login"))

    session.clear()
    session["user_id"] = user_record["user_id"]
    session["username"] = user_record["username"]
    session["role"] = user_record["role"]
    session["email"] = user_record["email"]

    flash(f"Welcome, {user_record['username']}! Signed in with Google.", "success")
    next_url = session.pop("next_url", None)
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        return redirect(next_url)
    return redirect(url_for("index"))


@app.route("/")
@login_required
def index():
    """Home page — upload form with patient details (optionally prefilled)."""
    prefill = None
    pid = request.args.get("patient_id", type=int)
    if pid:
        prefill = get_patient(pid)
    return render_template("index.html", prefill=prefill)


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """Process an uploaded retinal image through the full diagnostic pipeline."""

    # --- 1. Validate inputs ------------------------------------------------
    if "retinal_image" not in request.files:
        flash("No image file uploaded.", "danger")
        return redirect(url_for("index"))

    file = request.files["retinal_image"]
    if file.filename == "" or not _allowed_file(file.filename):
        flash("Please upload a valid retinal image (PNG, JPG, BMP, TIFF).", "danger")
        return redirect(url_for("index"))

    name = request.form.get("patient_name", "").strip()
    age_str = request.form.get("patient_age", "").strip()
    gender = request.form.get("patient_gender", "").strip()
    diabetes_history = request.form.get("diabetes_history", "").strip()

    if not name or not age_str or not gender:
        flash("Please fill in all required patient fields.", "danger")
        return redirect(url_for("index"))

    try:
        age = int(age_str)
    except ValueError:
        flash("Age must be a number.", "danger")
        return redirect(url_for("index"))

    today_str = datetime.now().strftime("%Y-%m-%d")
    upload_path: Path | None = None

    try:
        # --- 2. Save uploaded image ----------------------------------------
        safe_name = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{safe_name}"
        upload_path = UPLOAD_DIR / upload_filename
        file.save(str(upload_path))
        logger.info("Image saved: %s", upload_path)

        # Confirm the upload is a genuine, decodable image — the extension alone
        # is not enough (a renamed .txt would otherwise crash the pipeline).
        try:
            with Image.open(upload_path) as im:
                im.verify()
        except Exception:
            try:
                upload_path.unlink()
            except OSError:
                pass
            flash("The uploaded file is not a valid image.", "danger")
            return redirect(url_for("index"))

        # --- 3. Create or find patient -------------------------------------
        patient_id = add_patient(
            name=name,
            age=age,
            gender=gender,
            diabetes_history=diabetes_history if diabetes_history else "Not provided",
        )
        logger.info("Patient created/found with ID %d", patient_id)

        # --- 4. Model prediction -------------------------------------------
        prediction = predict_severity(str(upload_path))
        severity = prediction["severity"]
        confidence = prediction["confidence"]
        severity_label = prediction["severity_label"]
        model_used = prediction["model_used"]
        logger.info(
            "Prediction: %s (severity=%d, confidence=%.3f, model_used=%s)",
            severity_label, severity, confidence, model_used,
        )

        # --- 5. Grad-CAM heatmap -------------------------------------------
        model = load_model()
        heatmap_path = generate_gradcam(
            model=model,
            image_path=str(upload_path),
            predicted_class=severity,
            save_dir=str(HEATMAP_DIR),
        )
        logger.info("Heatmap saved: %s", heatmap_path)

        # --- 6. Treatment recommendation -----------------------------------
        recommendation = get_treatment_recommendation(severity)
        logger.info("Recommendation: %s", recommendation)

        # --- 7. Upload files to Supabase Storage & save report -------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_remote = f"uploads/{timestamp}_{upload_filename}"
        heat_remote = f"heatmaps/{timestamp}_{Path(heatmap_path).name}"

        upload_file(str(upload_path), BUCKET_IMAGES, img_remote)
        upload_file(heatmap_path, BUCKET_HEATMAPS, heat_remote)

        # Store bucket-prefixed paths so we can later generate signed URLs
        image_storage = f"{BUCKET_IMAGES}/{img_remote}"
        heatmap_storage = f"{BUCKET_HEATMAPS}/{heat_remote}"

        report_id = add_report(
            patient_id=patient_id,
            image_path=image_storage,
            severity=severity,
            confidence=confidence,
            heatmap_path=heatmap_storage,
            recommendation=recommendation,
            llm_summary="(pending)",
            date=today_str,
        )
        logger.info("Report saved with ID %d", report_id)

        # --- 8. Progression tracking ---------------------------------------
        progression_result = track_progression(patient_id, severity, visit_date=today_str)
        progression_status = progression_result["status"]
        logger.info("Progression status: %s", progression_status)

        # --- 9. Progression graph ------------------------------------------
        progression_graph_path = None
        try:
            progression_graph_path = plot_progression(patient_id)
            logger.info("Progression graph: %s", progression_graph_path)
        except ValueError as exc:
            logger.warning("Could not plot progression: %s", exc)

        # --- 10. Simulation (only if progression history exists) ------------
        simulation_path = None
        history = get_progression_history(patient_id)
        if history and len(history) > 0:
            try:
                sim_result = simulate_progression(
                    image_path=str(upload_path),
                    gradcam_path=heatmap_path,
                    progression_history=[dict(row) for row in history],
                )
                simulation_path = sim_result
                logger.info("Simulation saved: %s", simulation_path)
            except Exception as exc:
                logger.warning("Simulation failed: %s", exc)

        # --- 11. LLM explanation -------------------------------------------
        llm_summary = explain_report(severity, confidence, progression_status)
        logger.info("LLM explanation generated (%d chars).", len(llm_summary))

        # --- 12. Update the report with the LLM summary --------------------
        update_report_llm_summary(report_id, llm_summary)

        # --- 13. Generate PDF report ---------------------------------------
        patient_row = get_patient(patient_id)
        patient_dict = {
            "name": patient_row["name"],
            "age": patient_row["age"],
            "gender": patient_row["gender"],
            "diabetes_history": patient_row["diabetes_history"],
        }

        pdf_path = generate_pdf(
            report_id=report_id,
            patient=patient_dict,
            severity=severity,
            severity_label=severity_label,
            confidence=confidence,
            recommendation=recommendation,
            llm_summary=llm_summary,
            heatmap_path=heatmap_path,
            progression_graph_path=progression_graph_path,
            simulation_path=simulation_path,
            report_date=today_str,
            output_dir=str(REPORT_DIR),
        )
        logger.info("PDF report generated: %s", pdf_path)

        # Upload PDF to Supabase Storage
        pdf_remote = f"pdfs/report_{report_id}.pdf"
        upload_file(pdf_path, BUCKET_REPORTS, pdf_remote)

        return redirect(url_for("result", report_id=report_id))

    except Exception as exc:
        logger.exception("Upload pipeline failed: %s", exc)
        flash(f"Analysis failed: {exc}", "danger")
        return redirect(url_for("index"))


@app.route("/result/<int:report_id>")
@login_required
def result(report_id: int):
    """Display the full diagnostic result for a report."""
    conn = get_connection()
    try:
        report = conn.execute(
            "SELECT * FROM Reports WHERE report_id = ?", (report_id,)
        ).fetchone()

        if report is None:
            flash("Report not found.", "danger")
            return redirect(url_for("index"))

        patient = conn.execute(
            "SELECT * FROM Patients WHERE patient_id = ?",
            (report["patient_id"],),
        ).fetchone()

        # Get progression history for this patient
        progression = conn.execute(
            "SELECT * FROM Progression WHERE patient_id = ? ORDER BY date ASC",
            (report["patient_id"],),
        ).fetchall()

        # Determine latest progression status
        if progression:
            latest_status = progression[-1]["status"]
        else:
            latest_status = "Initial"

    finally:
        conn.close()

    severity = int(report["severity"])
    severity_label = config.DR_CLASSES.get(severity, "Unknown")
    confidence_pct = round(float(report["confidence"]) * 100, 1)

    # Generate signed URLs from Supabase Storage
    image_url   = storage_path_to_signed_url(report.get("image_path")) or ""
    heatmap_url = storage_path_to_signed_url(report.get("heatmap_path")) or ""

    # Progression graph — still served locally (generated at runtime)
    progression_graph = Path(config.BASE_DIR) / "reports" / f"progression_{report['patient_id']}.png"
    progression_graph_url = None
    if progression_graph.exists():
        progression_graph_url = url_for(
            "serve_file", subdir="reports",
            filename=f"progression_{report['patient_id']}.png",
        )

    # Simulation — still served locally
    simulation_url = None
    image_path = report.get("image_path", "")
    if image_path:
        sim_filename = f"{Path(str(image_path).split('/')[-1]).stem}_simulation.png"
        sim_path = Path(config.SIMULATION_DIR) / sim_filename
        if sim_path.exists():
            simulation_url = url_for("serve_file", subdir="simulations", filename=sim_filename)

    # Check for PDF in Supabase Storage or locally
    pdf_exists = (Path(config.REPORT_DIR) / f"report_{report_id}.pdf").exists()

    # A confidence of 0 means no real model ran (see model_loader fallback); the
    # template shows "N/A" rather than a fabricated 0.0% in that case.
    model_used = float(report["confidence"]) > 0

    return render_template(
        "result.html",
        report=report,
        patient=patient,
        severity=severity,
        severity_label=severity_label,
        confidence_pct=confidence_pct,
        recommendation=report["recommendation"],
        llm_summary=report["llm_summary"],
        progression_status=latest_status,
        progression=progression,
        image_url=image_url,
        heatmap_url=heatmap_url,
        progression_graph_url=progression_graph_url,
        simulation_url=simulation_url,
        pdf_exists=pdf_exists,
        model_used=model_used,
        dr_classes=config.DR_CLASSES,
    )



@app.route("/xai/<int:report_id>")
@login_required
def xai_explain(report_id: int):
    """Dedicated Explainable-AI dashboard for a single report.
    Runs all XAI components locally — no API key required.
    """
    conn = get_connection()
    try:
        report = conn.execute(
            "SELECT * FROM Reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        if report is None:
            flash("Report not found.", "danger")
            return redirect(url_for("dashboard"))
        patient = conn.execute(
            "SELECT * FROM Patients WHERE patient_id = ?",
            (report["patient_id"],),
        ).fetchone()
    finally:
        conn.close()

    # Resolve absolute paths for the XAI engine
    base = Path(config.BASE_DIR)
    image_rel = str(report["image_path"]).replace("\\", "/")
    heatmap_rel = str(report["heatmap_path"]).replace("\\", "/")

    # Try several path resolution strategies (same logic as serve_file)
    def _resolve(rel):
        for candidate in [
            base / rel,
            Path(config.STATIC_DIR) / Path(rel).name,
            Path(config.UPLOAD_DIR) / Path(rel).name,
            Path(config.HEATMAP_DIR) / Path(rel).name,
        ]:
            if candidate.exists():
                return str(candidate)
        return None

    abs_image = _resolve(image_rel)
    abs_heatmap = _resolve(heatmap_rel)

    if not abs_image:
        flash("Original image not found on disk — XAI requires the source image.", "warning")
        return redirect(url_for("result", report_id=report_id))

    severity = int(report["severity"])
    confidence = float(report["confidence"])

    # Run all XAI components
    xai = explain_prediction(
        image_path=abs_image,
        heatmap_path=abs_heatmap,
        severity=severity,
        confidence=confidence,
        probabilities=None,
    )

    image_url = db_path_to_url(report["image_path"])
    heatmap_url = db_path_to_url(report["heatmap_path"])

    return render_template(
        "xai.html",
        report=report,
        patient=patient,
        severity=severity,
        severity_label=config.DR_CLASSES.get(severity, "Unknown"),
        confidence_pct=round(confidence * 100, 1),
        image_url=image_url,
        heatmap_url=heatmap_url,
        xai=xai,
        dr_classes=config.DR_CLASSES,
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """Doctor dashboard — list all patients and their reports."""
    conn = get_connection()
    try:
        patients = conn.execute(
            "SELECT * FROM Patients ORDER BY patient_id DESC"
        ).fetchall()

        patient_data = []
        for p in patients:
            reports = conn.execute(
                "SELECT * FROM Reports WHERE patient_id = ? ORDER BY date DESC",
                (p["patient_id"],),
            ).fetchall()

            progression = conn.execute(
                "SELECT * FROM Progression WHERE patient_id = ? ORDER BY date DESC",
                (p["patient_id"],),
            ).fetchall()

            latest_status = progression[0]["status"] if progression else "No scans"
            latest_severity = None
            if reports:
                latest_severity = int(reports[0]["severity"])

            patient_data.append({
                "patient": p,
                "reports": reports,
                "progression": progression,
                "latest_status": latest_status,
                "latest_severity": latest_severity,
                "latest_severity_label": config.DR_CLASSES.get(latest_severity, "N/A") if latest_severity is not None else "N/A",
            })
    finally:
        conn.close()

    stats = {
        "patients": len(patient_data),
        "reports": sum(len(i["reports"]) for i in patient_data),
        "high_risk": sum(1 for i in patient_data if (i["latest_severity"] or 0) >= 3),
        "worsened": sum(1 for i in patient_data if i["latest_status"] == "Worsened"),
    }
    return render_template(
        "dashboard.html",
        patient_data=patient_data,
        dr_classes=config.DR_CLASSES,
        stats=stats,
    )


@app.route("/patient/<int:patient_id>")
@login_required
def patient_detail(patient_id: int):
    """Per-patient page: trend graph, progression timeline and scan history."""
    patient = get_patient(patient_id)
    if patient is None:
        flash("Patient not found.", "danger")
        return redirect(url_for("dashboard"))
    reports = list(reversed(get_reports_for_patient(patient_id)))
    progression = get_progression_history(patient_id)
    graph_url = None
    if progression:
        try:
            plot_progression(patient_id)
        except Exception as exc:
            logger.warning("Could not plot progression: %s", exc)
        if (Path(config.BASE_DIR) / "reports" / f"progression_{patient_id}.png").exists():
            graph_url = url_for("serve_file", subdir="reports", filename=f"progression_{patient_id}.png")
    return render_template(
        "patient.html",
        patient=patient,
        reports=reports,
        progression=progression,
        graph_url=graph_url,
        dr_classes=config.DR_CLASSES,
    )


@app.route("/report/<int:report_id>/regenerate", methods=["POST"])
@login_required
def regenerate_report(report_id: int):
    """Rebuild the PDF for an existing report."""
    conn = get_connection()
    try:
        report = conn.execute("SELECT * FROM Reports WHERE report_id = ?", (report_id,)).fetchone()
    finally:
        conn.close()
    if report is None:
        flash("Report not found.", "danger")
        return redirect(url_for("dashboard"))
    patient = get_patient(report["patient_id"])
    severity = int(report["severity"])
    base = Path(config.BASE_DIR)
    graph = base / "reports" / f"progression_{report['patient_id']}.png"
    sim = Path(config.SIMULATION_DIR) / f"{Path(str(report['image_path'])).stem}_simulation.png"
    try:
        generate_pdf(
            report_id=report_id,
            patient={k: patient[k] for k in ("name", "age", "gender", "diabetes_history")},
            severity=severity,
            severity_label=config.DR_CLASSES.get(severity, "Unknown"),
            confidence=float(report["confidence"]),
            recommendation=report["recommendation"],
            llm_summary=report["llm_summary"],
            heatmap_path=str(base / str(report["heatmap_path"]).replace("\\", "/")),
            progression_graph_path=str(graph) if graph.exists() else None,
            simulation_path=str(sim) if sim.exists() else None,
            report_date=str(report["date"]),
            output_dir=str(REPORT_DIR),
        )
        flash("PDF report regenerated.", "success")
    except Exception as exc:
        logger.exception("PDF regeneration failed")
        flash(f"Could not regenerate PDF: {exc}", "danger")
    return redirect(url_for("result", report_id=report_id))


@app.route("/download/<int:report_id>")
@login_required
def download_report(report_id: int):
    """Download the generated PDF report."""
    pdf_path = Path(config.REPORT_DIR) / f"report_{report_id}.pdf"
    if not pdf_path.exists():
        # Try the non-static reports dir as well
        pdf_path = Path(config.BASE_DIR) / "reports" / f"report_{report_id}.pdf"

    if not pdf_path.exists():
        flash("PDF report not found. Please regenerate.", "danger")
        return redirect(url_for("result", report_id=report_id))

    return send_file(
        str(pdf_path),
        as_attachment=True,
        download_name=f"NammaRetina_Report_{report_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/files/<subdir>/<filename>")
@login_required
def serve_file(subdir: str, filename: str):
    """Serve files — tries Supabase Storage signed URL first, then local fallback."""
    from flask import Response, redirect as flask_redirect
    safe_filename = secure_filename(filename)

    # Supabase Storage bucket mapping
    bucket_map = {
        "uploads": BUCKET_IMAGES,
        "heatmaps": BUCKET_HEATMAPS,
        "reports": BUCKET_REPORTS,
        "simulations": BUCKET_REPORTS,
    }
    bucket = bucket_map.get(subdir)
    if bucket:
        signed = get_signed_url(bucket, f"{subdir}/{safe_filename}")
        if signed:
            return flask_redirect(signed)

    # Local fallback (progression graphs, simulations still on disk)
    for search_dir in [
        Path(config.STATIC_DIR) / subdir,
        Path(config.BASE_DIR) / subdir,
        Path(config.UPLOAD_DIR),
        Path(config.HEATMAP_DIR),
        Path(config.REPORT_DIR),
        Path(config.BASE_DIR) / "reports",
    ]:
        p = Path(search_dir) / safe_filename
        if p.exists() and p.is_file():
            return send_file(str(p))

    # SVG placeholder
    logger.warning("File not found: %s/%s", subdir, filename)
    label = "Retinal Scan" if "heatmap" not in filename.lower() else "Grad-CAM Heatmap"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">'
           f'<rect width="300" height="300" fill="#1a365d"/>'
           f'<circle cx="150" cy="150" r="100" fill="#2a4365" stroke="#4299e1" stroke-width="2"/>'
           f'<text x="150" y="150" font-family="sans-serif" font-size="14" fill="#ebf8ff" '
           f'text-anchor="middle">{label}</text></svg>')
    return Response(svg, mimetype="image/svg+xml")


# ─── Initialisation ──────────────────────────────────────────────────────────

def create_app():
    """Application factory for external WSGI servers."""
    init_db()
    load_model()  # Pre-load model on startup
    return app


def _bootstrap() -> None:
    """Connect DB + warm the model. Runs under gunicorn and `python app.py`."""
    init_db()
    logger.info("Supabase database connected.")
    load_model()


# Gunicorn imports this module — run bootstrap at import time in production.
if os.getenv("SKIP_BOOTSTRAP", "").lower() not in {"1", "true", "yes"}:
    try:
        _bootstrap()
    except Exception as exc:
        # Allow the process to start so health checks / logs can surface the error.
        logger.error("Bootstrap failed: %s", exc)


if __name__ == "__main__":
    logger.info("Starting NammaRetina Flask application...")
    port = int(os.getenv("PORT", 5001))
    logger.info("Serving on http://127.0.0.1:%d", port)
    app.run(debug=config.DEBUG, host="0.0.0.0", port=port)
