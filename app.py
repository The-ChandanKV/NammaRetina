"""
NammaRetina - Flask Web Application (Phases 10-11)
Central web application wiring together all pipeline components:
    Upload -> Preprocess -> Model -> Grad-CAM -> Progression -> Simulation
    -> Treatment -> LLM -> DB -> PDF Report
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from database import (
    add_patient,
    add_report,
    get_connection,
    get_patient,
    get_progression_history,
    get_reports_for_patient,
    init_db,
    DB_PATH,
)
from gradcam import generate_gradcam
from llm_agent import explain_report
from model_loader import load_model, predict_severity
from progression import plot_progression, track_progression
from report_generator import generate_pdf
from simulation import simulate_progression
from treatment import get_treatment_recommendation

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


@app.route("/")
def index():
    """Home page — upload form with patient details."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
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

    # --- 2. Save uploaded image --------------------------------------------
    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_filename = f"{timestamp}_{safe_name}"
    upload_path = UPLOAD_DIR / upload_filename
    file.save(str(upload_path))
    logger.info("Image saved: %s", upload_path)

    # --- 3. Create or find patient -----------------------------------------
    patient_id = add_patient(
        name=name,
        age=age,
        gender=gender,
        diabetes_history=diabetes_history if diabetes_history else "Not provided",
    )
    logger.info("Patient created/found with ID %d", patient_id)

    # --- 4. Model prediction -----------------------------------------------
    prediction = predict_severity(str(upload_path))
    severity = prediction["severity"]
    confidence = prediction["confidence"]
    severity_label = prediction["severity_label"]
    model_used = prediction["model_used"]
    logger.info(
        "Prediction: %s (severity=%d, confidence=%.3f, model_used=%s)",
        severity_label, severity, confidence, model_used,
    )

    # --- 5. Grad-CAM heatmap -----------------------------------------------
    model = load_model()
    heatmap_path = generate_gradcam(
        model=model,
        image_path=str(upload_path),
        predicted_class=severity,
        save_dir=str(HEATMAP_DIR),
    )
    logger.info("Heatmap saved: %s", heatmap_path)

    # --- 6. Treatment recommendation ---------------------------------------
    recommendation = get_treatment_recommendation(severity)
    logger.info("Recommendation: %s", recommendation)

    # --- 7. Save the report to DB first (progression needs it) -------------
    # Store relative paths for portability in the DB
    image_rel = os.path.relpath(str(upload_path), config.BASE_DIR)
    heatmap_rel = os.path.relpath(heatmap_path, config.BASE_DIR)

    # We need to save the report BEFORE tracking progression because
    # track_progression reads the reports table to determine previous severity
    report_id = add_report(
        patient_id=patient_id,
        image_path=image_rel,
        severity=severity,
        confidence=confidence,
        heatmap_path=heatmap_rel,
        recommendation=recommendation,
        llm_summary="(pending)",  # Will update after LLM call
        date=today_str,
    )
    logger.info("Report saved with ID %d", report_id)

    # --- 8. Progression tracking -------------------------------------------
    progression_result = track_progression(patient_id, severity)
    progression_status = progression_result["status"]
    logger.info("Progression status: %s", progression_status)

    # --- 9. Progression graph ----------------------------------------------
    progression_graph_path = None
    try:
        progression_graph_path = plot_progression(patient_id)
        logger.info("Progression graph: %s", progression_graph_path)
    except ValueError as exc:
        logger.warning("Could not plot progression: %s", exc)

    # --- 10. Simulation (only if progression history exists) ----------------
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

    # --- 11. LLM explanation -----------------------------------------------
    llm_summary = explain_report(severity, confidence, progression_status)
    logger.info("LLM explanation generated (%d chars).", len(llm_summary))

    # --- 12. Update the report with the LLM summary ------------------------
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE Reports SET llm_summary = ? WHERE report_id = ?",
            (llm_summary, report_id),
        )
        conn.commit()
    finally:
        conn.close()

    # --- 13. Generate PDF report -------------------------------------------
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

    return redirect(url_for("result", report_id=report_id))


@app.route("/result/<int:report_id>")
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

    # Build paths for images — convert relative DB paths to static-servable paths
    heatmap_path = report["heatmap_path"]
    image_path = report["image_path"]

    # Check for progression graph
    progression_graph = Path(config.BASE_DIR) / "reports" / f"progression_{report['patient_id']}.png"
    progression_graph_url = None
    if progression_graph.exists():
        progression_graph_url = url_for(
            "serve_file", subdir="reports",
            filename=f"progression_{report['patient_id']}.png",
        )

    # Check for simulation
    simulation_url = None
    sim_path = Path(config.BASE_DIR) / "reports" / "simulated_progression.png"
    if sim_path.exists():
        simulation_url = url_for(
            "serve_file", subdir="reports", filename="simulated_progression.png"
        )

    # Check for PDF
    pdf_exists = (Path(config.REPORT_DIR) / f"report_{report_id}.pdf").exists()

    image_url = db_path_to_url(image_path)
    heatmap_url = db_path_to_url(heatmap_path)

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
        model_used=True,
    )


@app.route("/dashboard")
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

    return render_template(
        "dashboard.html",
        patient_data=patient_data,
        dr_classes=config.DR_CLASSES,
    )


@app.route("/download/<int:report_id>")
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
def serve_file(subdir: str, filename: str):
    """Serve uploaded images, heatmaps, and reports from project directories.

    Supports serving from both static/ subdirectories and top-level directories.
    Returns a clean SVG placeholder image if the file is missing instead of a redirect.
    """
    safe_filename = secure_filename(filename)

    # 1. Try static directory
    static_path = Path(config.STATIC_DIR) / subdir / safe_filename
    if static_path.exists() and static_path.is_file():
        return send_file(str(static_path))

    # 2. Try top-level project directories (uploads/, heatmaps/, reports/)
    project_path = Path(config.BASE_DIR) / subdir / safe_filename
    if project_path.exists() and project_path.is_file():
        return send_file(str(project_path))

    # 3. Try finding by filename in any of the standard image directories
    for search_dir in [config.UPLOAD_DIR, config.HEATMAP_DIR, config.REPORT_DIR, Path(config.BASE_DIR) / "uploads", Path(config.BASE_DIR) / "heatmaps", Path(config.BASE_DIR) / "reports"]:
        alt_path = Path(search_dir) / safe_filename
        if alt_path.exists() and alt_path.is_file():
            return send_file(str(alt_path))

    # 4. Fallback: return an SVG placeholder image instead of breaking with a 302 redirect
    logger.warning("File not found: %s/%s — serving SVG placeholder", subdir, filename)
    label = "Retinal Scan" if "heatmap" not in filename.lower() else "Grad-CAM Heatmap"
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
        <rect width="300" height="300" fill="#1a365d"/>
        <circle cx="150" cy="150" r="100" fill="#2a4365" stroke="#4299e1" stroke-width="2"/>
        <circle cx="150" cy="150" r="40" fill="#3182ce" opacity="0.6"/>
        <text x="150" y="145" font-family="sans-serif" font-size="14" fill="#ebf8ff" text-anchor="middle" font-weight="bold">{label}</text>
        <text x="150" y="165" font-family="sans-serif" font-size="11" fill="#90cdf4" text-anchor="middle">Archive Record</text>
    </svg>'''
    from flask import Response
    return Response(svg_content, mimetype="image/svg+xml")


# ─── Initialisation ──────────────────────────────────────────────────────────

def create_app():
    """Application factory for external WSGI servers."""
    init_db()
    load_model()  # Pre-load model on startup
    return app


if __name__ == "__main__":
    init_db()
    logger.info("Database initialised at %s", DB_PATH)

    load_model()  # Pre-load on startup

    logger.info("Starting NammaRetina Flask application...")
    app.run(debug=True, host="0.0.0.0", port=5000)
