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
import secrets
import json
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
    send_from_directory,
    session,
    url_for,
    abort,
)
from werkzeug.utils import secure_filename
from PIL import Image
import httpx
from postgrest.exceptions import APIError

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
    download_storage_file,
    get_report,
    get_all_reports,
    DatabaseUnavailable,
    storage_object_exists,
)
from evaluation_reports import ensure_seed_evaluation, latest_evaluation, list_evaluation_runs, load_run
from gradcam import generate_gradcam
from lime_explainer import explain_with_loaded_model, load_lime_artifact, save_lime_artifact
from llm_agent import explain_report
from model_loader import load_model, predict_severity
from progression import plot_progression, track_progression
from progression_analysis import analyse_longitudinal_progression
from report_generator import generate_pdf
from simulation import METHOD_DESCRIPTION as SIMULATION_METHOD_DESCRIPTION
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
app.secret_key = secrets.token_hex(32) if config.SECRET_KEY_IS_DEFAULT else config.SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.getenv("RENDER"))
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["TEMPLATES_AUTO_RELOAD"] = True  # always read templates fresh from disk


@app.after_request
def response_headers(response):
    response.headers["X-App-Revision"] = os.getenv("RENDER_GIT_COMMIT", "local")
    # Landing assets set their own Cache-Control; do not overwrite with no-store.
    if request.endpoint not in {"static", "landing_assets", "landing"}:
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response

if config.SECRET_KEY_IS_DEFAULT:
    logger.warning(
        "SECRET_KEY is the built-in development default. Set the SECRET_KEY "
        "environment variable to a random secret before deploying — sessions "
        "A temporary random session key is being used for this process."
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

# Public landing-page build output (Vite → static/landing/). Never put PHI here.
LANDING_DIR = Path(config.STATIC_DIR) / "landing"


@app.before_request
def protect_requests():
    # Flask's built-in static endpoint must never expose patient artifacts.
    # Landing assets are served only via /landing/* (see landing_assets).
    if request.endpoint == "static":
        filename = request.view_args.get("filename", "")
        if filename.startswith("landing/"):
            abort(404)
        if not filename.startswith("css/") or ".." in Path(filename).parts:
            abort(404)
    if request.method == "POST":
        expected = session.get("csrf_token", "")
        supplied = request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            abort(400, description="Your form expired. Reload the page and try again.")


@app.route("/health")
def health():
    from model_loader import is_model_available, model_fingerprint
    try:
        init_db()
        database_available = True
    except Exception:
        database_available = False
    model_available = is_model_available()
    ready = database_available and model_available
    return {"status": "ok" if ready else "unavailable", "model_available": model_available,
            "model_sha256": model_fingerprint(),
            "database_available": database_available,
            "revision": os.getenv("RENDER_GIT_COMMIT", "local")}, 200 if ready else 503


@app.errorhandler(httpx.RequestError)
@app.errorhandler(APIError)
@app.errorhandler(DatabaseUnavailable)
def backend_unavailable(error):
    logger.error("Backend service unavailable: %s", type(error).__name__)
    return render_template("unavailable.html"), 503


@app.context_processor
def form_tokens():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return {"csrf_token": session["csrf_token"]}


def resolve_artifact(storage_path, directory):
    """Resolve legacy files locally or restore a Supabase artifact after restart."""
    if not storage_path or str(storage_path) == "None":
        return None
    normalised = str(storage_path).replace("\\", "/")
    name = Path(normalised).name
    target = Path(directory) / name
    for candidate in (target, BASE_DIR / normalised):
        if candidate.is_file() and candidate.resolve().is_relative_to(BASE_DIR.resolve()):
            return str(candidate)
    if normalised.split("/", 1)[0] in {BUCKET_IMAGES, BUCKET_HEATMAPS, BUCKET_REPORTS}:
        try:
            return download_storage_file(normalised, target)
        except Exception:
            logger.warning("Could not restore artifact from storage", exc_info=True)
    return None


def simulation_filename_for_image(image_path: str | None) -> str | None:
    """Derive the on-disk / storage simulation filename from a report image path."""
    if not image_path:
        return None
    sim_filename = f"{Path(str(image_path).split('/')[-1]).stem}_simulation.png"
    # Older uploads used a second timestamp only in their remote object name.
    if len(sim_filename) > 32 and sim_filename[:8].isdigit() and sim_filename[16:24].isdigit():
        sim_filename = sim_filename[16:]
    return sim_filename


def ensure_simulation_for_report(report: dict, progression_rows: list | None) -> str | None:
    """Return a /files/simulations/... URL only when a real simulation image exists.

    Regenerates from the retinal scan + Grad-CAM when the artifact is missing,
    then persists it to local disk and Supabase Storage when possible.
    """
    image_path = report.get("image_path")
    heatmap_path = report.get("heatmap_path")
    sim_filename = simulation_filename_for_image(image_path)
    if not sim_filename or not heatmap_path:
        return None

    local_sim = Path(config.SIMULATION_DIR) / sim_filename
    remote_key = f"simulations/{sim_filename}"

    if local_sim.is_file() and local_sim.stat().st_size > 0:
        return url_for("serve_file", subdir="simulations", filename=sim_filename)

    # Prefer an already-uploaded simulation without regenerating.
    if get_signed_url(BUCKET_REPORTS, remote_key):
        try:
            download_storage_file(f"{BUCKET_REPORTS}/{remote_key}", local_sim)
        except Exception:
            logger.info("Using remote simulation URL without local cache: %s", remote_key)
        return url_for("serve_file", subdir="simulations", filename=sim_filename)

    image_local = resolve_artifact(image_path, UPLOAD_DIR)
    heatmap_local = resolve_artifact(heatmap_path, HEATMAP_DIR)
    if not image_local or not heatmap_local:
        logger.warning(
            "Cannot regenerate simulation for report %s — missing image/heatmap artifacts",
            report.get("report_id"),
        )
        return None

    history = [dict(row) for row in (progression_rows or [])]
    if not history:
        history = [{
            "status": "Initial",
            "current_severity": int(report.get("severity") or 0),
            "date": str(report.get("date") or ""),
        }]

    try:
        sim_path = simulate_progression(
            image_path=image_local,
            gradcam_path=heatmap_local,
            progression_history=history,
        )
        try:
            upload_file(sim_path, BUCKET_REPORTS, remote_key)
        except Exception:
            logger.warning("Simulation generated but storage upload failed", exc_info=True)
        return url_for("serve_file", subdir="simulations", filename=Path(sim_path).name)
    except Exception:
        logger.warning("Simulation regeneration failed", exc_info=True)
        return None


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def db_path_to_url(rel_path: str | None) -> str:
    """Convert a DB-stored relative path like 'static\\uploads\\file.png' into a URL."""
    if not rel_path:
        return url_for("serve_file", subdir="placeholder", filename="none.svg")
    if str(rel_path).split("/", 1)[0] in {BUCKET_IMAGES, BUCKET_HEATMAPS, BUCKET_REPORTS}:
        return storage_path_to_signed_url(rel_path) or ""
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
    """Redirect anonymous users to /signin, preserving the intended destination."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("signin", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    """Restrict research/evaluation pages to administrator accounts."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("signin", next=request.path))
        if session.get("role") != "admin":
            flash("Research evaluation is restricted to administrator accounts.", "warning")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    """Create a new account. The first account created becomes the admin."""
    first_account = count_users() == 0
    if not first_account and session.get("role") != "admin":
        flash("Account registration is restricted. Ask your administrator for access.", "warning")
        return redirect(url_for("signin"))
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

        role = "admin" if first_account else "doctor"
        try:
            user_id = create_user(username, password, email=email, role=role)
        except Exception as exc:
            flash(f"Registration failed: {exc}", "danger")
            return redirect(url_for("register"))

        if not first_account:
            flash(f"Account created for {username}.", "success")
            return redirect(url_for("dashboard"))
        session.clear()
        session["user_id"] = user_id
        session["username"] = username
        session["role"] = role
        flash(f"Welcome, {username}! Your account has been created.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html", first_account=first_account)


def _safe_next_url(default_endpoint: str = "dashboard") -> str:
    """Return a same-origin relative next URL, else the default doctor destination."""
    next_url = request.args.get("next") or request.form.get("next") or ""
    if next_url.startswith("/") and not next_url.startswith("//"):
        # Keep auth pages from bouncing signed-in users back into the login shell.
        if next_url.rstrip("/") in {"/login", "/signin"}:
            return url_for(default_endpoint)
        return next_url
    return url_for(default_endpoint)


def _auth_signin_view():
    """Shared email/password sign-in handler for /signin and /login."""
    if session.get("user_id") and request.method == "GET":
        return redirect(_safe_next_url("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = verify_user(email, password)
        if user is None:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("signin"))

        session.clear()
        session["user_id"] = user["user_id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(_safe_next_url("dashboard"))

    return render_template("login.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    """Primary public sign-in page (no doctor sidebar)."""
    return _auth_signin_view()


@app.route("/login", methods=["GET", "POST"])
def login():
    """Back-compat alias for /signin."""
    return _auth_signin_view()


@app.route("/logout", methods=["POST"])
def logout():
    """Clear the session and return to the public landing page."""
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("landing"))


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
        return redirect(url_for("signin"))


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
        return redirect(url_for("signin"))

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
            return redirect(url_for("signin"))
    elif access_token:
        try:
            user = get_user_from_access_token(access_token)
        except Exception as exc:
            logger.error("Failed to retrieve user from access token: %s", exc)
            flash(f"Authentication failed: {exc}", "danger")
            return redirect(url_for("signin"))

    if not user:
        flash("Could not retrieve your user details from Google.", "danger")
        return redirect(url_for("signin"))

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
        return redirect(url_for("signin"))

    next_url = session.get("next_url")
    session.clear()
    session["user_id"] = user_record["user_id"]
    session["username"] = user_record["username"]
    session["role"] = user_record["role"]
    session["email"] = user_record["email"]

    flash(f"Welcome, {user_record['username']}! Signed in with Google.", "success")
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        return redirect(next_url)
    return redirect(url_for("dashboard"))


@app.route("/privacy")
def privacy_policy():
    """Public privacy policy for OAuth / App Store consent requirements."""
    return render_template("privacy.html")


@app.route("/")
def landing():
    """Public Vite landing page (no authentication)."""
    index_html = LANDING_DIR / "index.html"
    if not index_html.is_file():
        logger.error("Landing build missing at %s — run: cd landing-source && npm run build", LANDING_DIR)
        flash(
            "Landing page assets are not built yet. Open /login to continue, "
            "or build with: cd landing-source && npm install && npm run build",
            "warning",
        )
        return redirect(url_for("signin"))
    response = send_from_directory(LANDING_DIR, "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/favicon.ico")
def favicon():
    """Browsers request /favicon.ico by default — serve the NammaRetina mark."""
    icon = LANDING_DIR / "favicon.ico"
    if not icon.is_file():
        abort(404)
    response = send_from_directory(LANDING_DIR, "favicon.ico")
    response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
    return response


@app.route("/landing/<path:filename>")
def landing_assets(filename: str):
    """Serve only compiled public landing assets (JS/CSS/images/fonts)."""
    if ".." in Path(filename).parts:
        abort(404)
    target = (LANDING_DIR / filename).resolve()
    try:
        target.relative_to(LANDING_DIR.resolve())
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    response = send_from_directory(LANDING_DIR, filename)
    # Favicons must not be immutable-cached (browsers retain stale tab icons).
    name = Path(filename).name.lower()
    if name.startswith("favicon") or name.endswith(".ico"):
        response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
    else:
        # Fingerprinted Vite assets (JS/CSS/images with hashes) can be cached long-term.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@app.route("/scan")
@login_required
def index():
    """Authenticated retinal upload form (endpoint name kept as ``index``)."""
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

    if not 0 <= age <= 150 or gender not in {"Male", "Female", "Other"}:
        flash("Please provide a valid age and gender.", "danger")
        return redirect(url_for("index"))

    patient_id = request.form.get("patient_id", type=int)
    if request.form.get("patient_id") and (not patient_id or not get_patient(patient_id)):
        abort(404)

    today_str = datetime.now().strftime("%Y-%m-%d")
    upload_path: Path | None = None

    try:
        # --- 2. Save uploaded image ----------------------------------------
        safe_name = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{secrets.token_hex(12)}_{safe_name}"
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

        # --- 4. Model prediction -------------------------------------------
        prediction = predict_severity(str(upload_path))
        if not prediction["model_used"]:
            upload_path.unlink(missing_ok=True)
            flash("Analysis unavailable: the trained model is not loaded. No diagnosis or visit was recorded.", "warning")
            return redirect(url_for("index"))
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

        if not patient_id:
            patient_id = add_patient(name=name, age=age, gender=gender,
                                     diabetes_history=diabetes_history or "Not provided")

        # --- 6. Treatment recommendation -----------------------------------
        recommendation = get_treatment_recommendation(severity)
        logger.info("Recommendation: %s", recommendation)

        # --- 7. Upload files to Supabase Storage & save report -------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_remote = f"uploads/{upload_filename}"
        heat_remote = f"heatmaps/{Path(heatmap_path).name}" if heatmap_path else None

        upload_file(str(upload_path), BUCKET_IMAGES, img_remote)
        if heatmap_path:
            upload_file(heatmap_path, BUCKET_HEATMAPS, heat_remote)

        # Store bucket-prefixed paths so we can later generate signed URLs
        image_storage = f"{BUCKET_IMAGES}/{img_remote}"
        heatmap_storage = f"{BUCKET_HEATMAPS}/{heat_remote}" if heat_remote else ""

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
        progression_result = track_progression(patient_id, severity, visit_date=today_str, report_id=report_id)
        progression_status = progression_result["status"]
        logger.info("Progression status: %s", progression_status)

        # --- 9. Progression graph ------------------------------------------
        progression_graph_path = None
        try:
            progression_graph_path = plot_progression(patient_id)
            logger.info("Progression graph: %s", progression_graph_path)
        except ValueError as exc:
            logger.warning("Could not plot progression: %s", exc)

        # --- 10. Disease-spread simulation (Grad-CAM + progression trend) ---
        simulation_path = None
        history = get_progression_history(patient_id)
        if heatmap_path and history:
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

        if simulation_path:
            try:
                upload_file(simulation_path, BUCKET_REPORTS, f"simulations/{Path(simulation_path).name}")
            except Exception:
                logger.warning("Simulation persistence unavailable", exc_info=True)
        probabilities_path = REPORT_DIR / f"probabilities_{report_id}.json"
        probabilities_path.write_text(json.dumps(prediction["probabilities"]))
        try:
            upload_file(str(probabilities_path), BUCKET_REPORTS, f"metadata/{probabilities_path.name}")
        except Exception:
            logger.warning("Probability metadata persistence unavailable", exc_info=True)

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

        # A historical report describes the change at that visit, not today's latest visit.
        reports = get_reports_for_patient(report["patient_id"])
        prior = [r for r in reports if (str(r["date"]), r["report_id"]) <
                 (str(report["date"]), report_id) and float(r.get("confidence") or 0) > 0]
        latest_status = "Initial"
        if prior and float(report.get("confidence") or 0) > 0:
            difference = int(report["severity"]) - int(prior[-1]["severity"])
            latest_status = "Worsened" if difference > 0 else "Improved" if difference < 0 else "Stable"

    finally:
        conn.close()

    severity = int(report["severity"])
    severity_label = config.DR_CLASSES.get(severity, "Unknown")
    confidence_pct = round(float(report["confidence"]) * 100, 1)

    # Only expose media URLs when the underlying artifact actually exists.
    image_path = report.get("image_path")
    heatmap_path = report.get("heatmap_path")
    image_available = bool(image_path) and (
        storage_object_exists(str(image_path))
        or bool(resolve_artifact(image_path, UPLOAD_DIR))
    )
    heatmap_available = bool(heatmap_path) and (
        storage_object_exists(str(heatmap_path))
        or bool(resolve_artifact(heatmap_path, HEATMAP_DIR))
    )
    image_url = db_path_to_url(image_path) if image_available else ""
    heatmap_url = db_path_to_url(heatmap_path) if heatmap_available else ""
    media_missing = not image_available

    # Progression graph — still served locally (generated at runtime)
    progression_graph = Path(config.BASE_DIR) / "reports" / f"progression_{report['patient_id']}.png"
    progression_graph_url = None
    if progression_graph.exists():
        progression_graph_url = url_for(
            "serve_file", subdir="reports",
            filename=f"progression_{report['patient_id']}.png",
        )

    # A confidence of 0 means no real model ran (see model_loader fallback); the
    # template shows "N/A" rather than a fabricated 0.0% in that case.
    model_used = float(report["confidence"]) > 0
    if not model_used:
        severity_label = "Analysis unavailable"
        latest_status = "Unavailable"

    # Incomplete / stub summaries should use the template fallback explanation.
    raw_summary = (report.get("llm_summary") or "").strip()
    summary_is_stub = raw_summary.lower() in {
        "placeholder", "(pending)", "pending", "", "none", "null",
    }

    # PDF actions only when prediction + source scan exist (regenerate can rebuild).
    pdf_exists = bool(model_used and image_available)

    # Simulation — restore/regenerate only when both scan + heatmap exist.
    simulation_url = None
    if model_used and image_available and heatmap_available:
        simulation_url = ensure_simulation_for_report(dict(report), list(progression or []))

    longitudinal = analyse_longitudinal_progression(
        list(get_reports_for_patient(report["patient_id"]) or []),
        patient=dict(patient) if patient else None,
    )

    return render_template(
        "result.html",
        report=report,
        patient=patient,
        severity=severity,
        severity_label=severity_label,
        confidence_pct=confidence_pct,
        recommendation=report["recommendation"] if model_used else "A new scan analysis with a loaded model is required.",
        llm_summary=(
            "" if summary_is_stub else
            (report["llm_summary"] if model_used else "No model prediction was made for this record. It cannot establish severity or progression.")
        ),
        progression_status=latest_status,
        progression=progression,
        longitudinal=longitudinal,
        simulation_method=SIMULATION_METHOD_DESCRIPTION,
        image_url=image_url,
        heatmap_url=heatmap_url,
        image_available=image_available,
        heatmap_available=heatmap_available,
        media_missing=media_missing,
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

    if float(report.get("confidence") or 0) <= 0:
        flash("No prediction is available to explain for this report.", "warning")
        return redirect(url_for("result", report_id=report_id))
    abs_image = resolve_artifact(report["image_path"], UPLOAD_DIR)
    abs_heatmap = resolve_artifact(report["heatmap_path"], HEATMAP_DIR)

    if not abs_image:
        flash("Original image not found on disk — XAI requires the source image.", "warning")
        return redirect(url_for("result", report_id=report_id))

    severity = int(report["severity"])
    confidence = float(report["confidence"])

    probabilities = None
    metadata = resolve_artifact(f"{BUCKET_REPORTS}/metadata/probabilities_{report_id}.json", REPORT_DIR)
    if metadata:
        try:
            probabilities = json.loads(Path(metadata).read_text())
        except (ValueError, OSError):
            logger.warning("Stored probability metadata is unavailable")
    # Run image analysis using only recorded model outputs.
    # Genuine LIME is on-demand (expensive) and loaded from cache when present.
    xai = explain_prediction(
        image_path=abs_image,
        heatmap_path=abs_heatmap,
        severity=severity,
        confidence=confidence,
        probabilities=probabilities,
    )
    lime_cached = load_lime_artifact(REPORT_DIR, report_id)
    lime_url = None
    if lime_cached and Path(lime_cached.get("overlay_path", "")).exists():
        lime_url = url_for("serve_file", subdir="reports", filename=Path(lime_cached["overlay_path"]).name)

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
        lime=lime_cached,
        lime_url=lime_url,
        dr_classes=config.DR_CLASSES,
    )


@app.route("/xai/<int:report_id>/lime", methods=["POST"])
@login_required
def generate_lime(report_id: int):
    """On-demand genuine LIME explanation for the predicted class."""
    report = get_report(report_id)
    if report is None:
        abort(404)
    if float(report.get("confidence") or 0) <= 0:
        flash("No prediction is available to explain for this report.", "warning")
        return redirect(url_for("result", report_id=report_id))
    abs_image = resolve_artifact(report["image_path"], UPLOAD_DIR)
    if not abs_image:
        flash("Original image not found — LIME requires the source image.", "warning")
        return redirect(url_for("xai_explain", report_id=report_id))
    try:
        n_samples = int(request.form.get("n_samples") or 160)
        n_samples = max(40, min(n_samples, 400))
        result = explain_with_loaded_model(
            abs_image,
            target_class=int(report["severity"]),
            n_samples=n_samples,
            n_segments=36,
            batch_size=16,
            seed=42,
        )
        save_lime_artifact(result, REPORT_DIR, report_id)
        flash("LIME explanation generated for the predicted class.", "success")
    except Exception as exc:
        logger.exception("LIME generation failed for report %s", report_id)
        flash(f"LIME explanation unavailable: {exc}", "warning")
    return redirect(url_for("xai_explain", report_id=report_id))


@app.route("/evaluation")
@admin_required
def evaluation_dashboard():
    """Administrator-only research evaluation dashboard (no patient PHI)."""
    evaluation = ensure_seed_evaluation()
    runs = [
        {"id": path.name, "metrics": load_run(path)}
        for path in list_evaluation_runs()[:12]
    ]
    return render_template(
        "evaluation.html",
        evaluation=evaluation or latest_evaluation(),
        runs=runs,
    )


def _patient_status_summary(
    latest_severity: int | None,
    latest_status: str,
    n_reports: int,
) -> str:
    """Short 3–4 word clinical status line for patient lists."""
    if n_reports <= 0 or latest_status == "No scans":
        return "Awaiting first scan"
    if latest_severity is None or latest_status == "Unavailable":
        return "Analysis not available"
    grade = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR",
    }.get(int(latest_severity), "Unknown grade")
    if latest_status == "Worsened":
        return f"{grade} — worsening"
    if latest_status == "Improved":
        return f"{grade} — improving"
    if latest_status == "Stable":
        return f"{grade} — stable"
    return f"{grade} — initial"


def _build_patient_directory() -> list[dict]:
    """Load all patients with latest severity, progression status, and one-liner."""
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
            if reports and float(reports[0].get("confidence") or 0) > 0:
                latest_severity = int(reports[0]["severity"])
            elif reports:
                latest_status = "Unavailable"
            patient_data.append({
                "patient": p,
                "reports": reports,
                "progression": progression,
                "latest_status": latest_status,
                "latest_severity": latest_severity,
                "latest_severity_label": (
                    config.DR_CLASSES.get(latest_severity, "N/A")
                    if latest_severity is not None else "N/A"
                ),
                "summary": _patient_status_summary(
                    latest_severity, latest_status, len(reports)
                ),
            })
        return patient_data
    finally:
        conn.close()


@app.route("/patients")
@login_required
def patients_list():
    """Directory of all patients with current status and a short summary."""
    patient_data = _build_patient_directory()
    return render_template(
        "patients.html",
        patient_data=patient_data,
        dr_classes=config.DR_CLASSES,
        total_patients=len(patient_data),
    )


@app.route("/reports")
@login_required
def reports_list():
    """List every diagnostic report across all patients."""
    reports = get_all_reports()
    rows = []
    patient_cache: dict[int, dict | None] = {}
    for report in reports:
        pid = int(report["patient_id"])
        if pid not in patient_cache:
            patient_cache[pid] = get_patient(pid)
        patient = patient_cache[pid]
        confidence = float(report.get("confidence") or 0)
        model_used = confidence > 0
        severity = int(report.get("severity") or 0) if model_used else None
        rows.append(
            {
                "report": report,
                "patient": patient,
                "model_used": model_used,
                "severity": severity,
                "severity_label": (
                    config.DR_CLASSES.get(severity, "Unknown") if model_used else "Analysis unavailable"
                ),
                "confidence_pct": round(confidence * 100, 1) if model_used else None,
            }
        )
    return render_template(
        "reports.html",
        report_rows=rows,
        dr_classes=config.DR_CLASSES,
        total_reports=len(rows),
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """Doctor dashboard — list all patients and their reports."""
    patient_data = _build_patient_directory()
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

    latest_report = reports[0] if reports else None
    latest_image_url = ""
    latest_heatmap_url = ""
    latest_pdf_ready = False
    if latest_report:
        latest_image_url = db_path_to_url(latest_report.get("image_path"))
        if latest_report.get("heatmap_path"):
            latest_heatmap_url = db_path_to_url(latest_report.get("heatmap_path"))
        latest_pdf_ready = float(latest_report.get("confidence") or 0) > 0

    # Chart geometry from real progression rows (oldest → newest).
    chart_points = []
    for row in progression or []:
        try:
            chart_points.append(
                {
                    "date": str(row.get("date") or ""),
                    "severity": int(row.get("current_severity") if row.get("current_severity") is not None else 0),
                    "status": row.get("status") or "",
                }
            )
        except (TypeError, ValueError):
            continue

    chart_width, chart_height = 320, 160
    pad_l, pad_r, pad_t, pad_b = 28, 16, 18, 28
    plot_w = chart_width - pad_l - pad_r
    plot_h = chart_height - pad_t - pad_b
    n = len(chart_points)
    chart_marks = []
    for idx, point in enumerate(chart_points):
        x = pad_l if n == 1 else pad_l + (idx / (n - 1)) * plot_w
        y = pad_t + plot_h - (point["severity"] / 4.0) * plot_h
        label = point["date"][5:] if len(point["date"]) >= 10 else point["date"]
        chart_marks.append({**point, "x": round(x, 2), "y": round(y, 2), "label": label})
    polyline = " ".join(f"{m['x']},{m['y']}" for m in chart_marks)
    area_path = ""
    if chart_marks:
        first, last = chart_marks[0], chart_marks[-1]
        area_path = (
            f"M{first['x']} {pad_t + plot_h} "
            + " ".join(f"L{m['x']} {m['y']}" for m in chart_marks)
            + f" L{last['x']} {pad_t + plot_h} Z"
        )

    longitudinal = analyse_longitudinal_progression(
        list(get_reports_for_patient(patient_id) or []),
        patient=dict(patient) if patient else None,
    )

    return render_template(
        "patient.html",
        patient=patient,
        reports=reports,
        progression=progression,
        longitudinal=longitudinal,
        graph_url=graph_url,
        dr_classes=config.DR_CLASSES,
        latest_report=latest_report,
        latest_image_url=latest_image_url,
        latest_heatmap_url=latest_heatmap_url,
        latest_pdf_ready=latest_pdf_ready,
        chart_marks=chart_marks,
        chart_polyline=polyline,
        chart_area_path=area_path,
        chart_width=chart_width,
        chart_height=chart_height,
        chart_pad_l=pad_l,
        chart_pad_r=pad_r,
        chart_pad_t=pad_t,
        chart_pad_b=pad_b,
        chart_plot_h=plot_h,
    )


@app.route("/report/<int:report_id>/regenerate", methods=["POST"])
@login_required
def regenerate_report(report_id: int):
    """Rebuild the PDF for an existing report."""
    report = get_report(report_id)
    if report is None:
        flash("Report not found.", "danger")
        return redirect(url_for("dashboard"))
    if float(report.get("confidence") or 0) <= 0:
        flash("This record has no model prediction. Please run a new analysis before generating a diagnostic PDF.", "warning")
        return redirect(url_for("result", report_id=report_id))
    try:
        pdf_path = _build_report_pdf(dict(report))
        if not pdf_path:
            flash("Could not regenerate PDF for this report.", "danger")
        else:
            flash("PDF report regenerated.", "success")
    except Exception as exc:
        logger.exception("PDF regeneration failed")
        flash(f"Could not regenerate PDF: {exc}", "danger")
    return redirect(url_for("result", report_id=report_id))


def _build_report_pdf(report: dict) -> str | None:
    """Create (or rebuild) the PDF for a report and persist it to Storage when possible."""
    if float(report.get("confidence") or 0) <= 0:
        return None
    patient = get_patient(report["patient_id"])
    if patient is None:
        return None
    severity = int(report["severity"])
    ensure_simulation_for_report(dict(report), list(get_progression_history(report["patient_id"]) or []))
    sim_name = simulation_filename_for_image(report.get("image_path"))
    sim = Path(config.SIMULATION_DIR) / sim_name if sim_name else None
    graph = Path(config.BASE_DIR) / "reports" / f"progression_{report['patient_id']}.png"
    heatmap = resolve_artifact(report.get("heatmap_path"), HEATMAP_DIR)
    pdf_path = generate_pdf(
        report_id=int(report["report_id"]),
        patient={k: patient[k] for k in ("name", "age", "gender", "diabetes_history")},
        severity=severity,
        severity_label=config.DR_CLASSES.get(severity, "Unknown"),
        confidence=float(report["confidence"]),
        recommendation=report.get("recommendation") or "",
        llm_summary=report.get("llm_summary") or "",
        heatmap_path=heatmap,
        progression_graph_path=str(graph) if graph.exists() else None,
        simulation_path=str(sim) if sim and sim.exists() else None,
        report_date=str(report.get("date") or ""),
        output_dir=str(REPORT_DIR),
    )
    try:
        upload_file(pdf_path, BUCKET_REPORTS, f"pdfs/report_{report['report_id']}.pdf")
    except Exception:
        logger.warning("PDF generated but storage upload failed for report %s", report.get("report_id"), exc_info=True)
    return pdf_path


@app.route("/download/<int:report_id>")
@login_required
def download_report(report_id: int):
    """Download the generated PDF report."""
    report = get_report(report_id)
    if report is None:
        abort(404)
    if float(report.get("confidence") or 0) <= 0:
        flash("This record has no valid model prediction. A diagnostic PDF is unavailable.", "warning")
        return redirect(url_for("result", report_id=report_id))
    pdf_path = Path(REPORT_DIR) / f"report_{report_id}.pdf"
    if not pdf_path.exists():
        restored = resolve_artifact(f"{BUCKET_REPORTS}/pdfs/report_{report_id}.pdf", REPORT_DIR)
        if restored:
            pdf_path = Path(restored)

    if not pdf_path.exists():
        # Try the non-static reports dir as well
        pdf_path = Path(config.BASE_DIR) / "reports" / f"report_{report_id}.pdf"

    if not pdf_path.exists():
        try:
            rebuilt = _build_report_pdf(dict(report))
            if rebuilt:
                pdf_path = Path(rebuilt)
        except Exception:
            logger.exception("Automatic PDF rebuild failed for report %s", report_id)

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
    if subdir not in {"uploads", "heatmaps", "reports", "simulations", "placeholder"}:
        abort(404)

    # Graphs are regenerated from current records and must be served locally first.
    if subdir == "reports" and safe_filename.startswith("progression_"):
        graph = BASE_DIR / "reports" / safe_filename
        if graph.is_file():
            return send_file(str(graph))

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

    # SVG placeholder — never implied to be a real simulation/heatmap.
    logger.warning("File not found: %s/%s", subdir, filename)
    lower = filename.lower()
    if subdir == "simulations" or "simulation" in lower:
        label = "Simulation unavailable"
    elif "heatmap" in lower or "gradcam" in lower:
        label = "Grad-CAM unavailable"
    else:
        label = "Image unavailable"
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
    try:
        init_db()
        logger.info("Supabase database connected.")
    finally:
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
