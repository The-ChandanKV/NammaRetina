"""
NammaRetina - Database Layer (Supabase)
=======================================
All data operations now go through the Supabase Python client using the
service-role key (bypasses RLS for server-side use).

Public API is identical to the old SQLite version so all callers
(app.py, progression.py, etc.) need zero changes to their call sites.
Rows are returned as plain dicts instead of sqlite3.Row objects.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

logger = logging.getLogger(__name__)

# ── Keep DB_PATH exported so any leftover imports don't crash ──────────────────
DB_PATH = None   # SQLite no longer used


# ── Supabase client (singleton) ───────────────────────────────────────────────

_sb: Client | None = None


def _get_sb() -> Client:
    """Return a Supabase client using the service-role key (server-side only)."""
    global _sb
    if _sb is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not url or not key or "YOUR_PROJECT_ID" in url:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env "
                "before starting the application."
            )
        _sb = create_client(url, key)
    return _sb


def get_connection():
    """Compatibility shim — returns the Supabase client instead of a sqlite3 connection.
    Code that calls conn.execute(...) directly in app.py has been updated to use
    the helper functions below instead.
    """
    return _get_sb()


def _next_id(table: str, id_col: str) -> int:
    """Return max(id)+1. Used when Postgres sequences are out of sync after migration."""
    sb = _get_sb()
    result = (
        sb.table(table)
        .select(id_col)
        .order(id_col, desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return 1
    return int(result.data[0][id_col]) + 1


def _is_duplicate_key(exc: Exception) -> bool:
    text = str(exc).lower()
    return "23505" in text or "duplicate key" in text


def _insert_with_id_fallback(table: str, id_col: str, payload: dict) -> dict:
    """Insert a row; on sequence/pkey clash, retry with an explicit next id."""
    sb = _get_sb()
    try:
        result = sb.table(table).insert(payload).execute()
        return result.data[0]
    except Exception as exc:
        if not _is_duplicate_key(exc):
            raise
        payload = dict(payload)
        payload[id_col] = _next_id(table, id_col)
        logger.warning(
            "Sequence for %s.%s looks stale; retrying insert with %s=%s",
            table, id_col, id_col, payload[id_col],
        )
        result = sb.table(table).insert(payload).execute()
        return result.data[0]


def init_db(*args, **kwargs) -> str:
    """Verify Supabase connectivity on startup. Tables are created via the SQL Editor."""
    try:
        sb = _get_sb()
        sb.table("patients").select("patient_id").limit(1).execute()
        logger.info("Supabase connection verified.")
        return "supabase"
    except Exception as exc:
        logger.error("Supabase connection failed: %s", exc)
        raise


# ── Patients ──────────────────────────────────────────────────────────────────

def add_patient(
    name: str,
    age: int,
    gender: str,
    diabetes_history: str,
    db_path=None,
) -> int:
    """Return patient_id, creating the patient only if they don't already exist."""
    sb = _get_sb()
    existing = (
        sb.table("patients")
        .select("patient_id")
        .ilike("name", name.strip())
        .eq("age", age)
        .ilike("gender", gender.strip())
        .limit(1)
        .execute()
    )
    if existing.data:
        return int(existing.data[0]["patient_id"])

    row = _insert_with_id_fallback(
        "patients",
        "patient_id",
        {
            "name": name.strip(),
            "age": age,
            "gender": gender.strip(),
            "diabetes_history": diabetes_history,
        },
    )
    return int(row["patient_id"])


def get_patient(patient_id: int, db_path=None) -> Optional[dict]:
    """Fetch a single patient by patient_id. Returns a dict or None."""
    sb = _get_sb()
    result = (
        sb.table("patients")
        .select("*")
        .eq("patient_id", patient_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ── Reports ───────────────────────────────────────────────────────────────────

def add_report(
    patient_id: int,
    image_path: str,
    severity: int,
    confidence: float,
    heatmap_path: str,
    recommendation: str,
    llm_summary: str,
    date: Optional[str] = None,
    db_path=None,
) -> int:
    """Insert a report row and return the new report_id."""
    if date is None:
        date = datetime.date.today().isoformat()

    row = _insert_with_id_fallback(
        "reports",
        "report_id",
        {
            "patient_id": patient_id,
            "image_path": image_path,
            "severity": severity,
            "confidence": confidence,
            "heatmap_path": heatmap_path,
            "recommendation": recommendation,
            "llm_summary": llm_summary,
            "date": date,
        },
    )
    return int(row["report_id"])


def get_report(report_id: int) -> Optional[dict]:
    """Fetch a single report by report_id."""
    sb = _get_sb()
    result = (
        sb.table("reports")
        .select("*")
        .eq("report_id", report_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def update_report_llm_summary(report_id: int, llm_summary: str) -> None:
    """Update the llm_summary field of an existing report."""
    sb = _get_sb()
    sb.table("reports").update({"llm_summary": llm_summary}).eq("report_id", report_id).execute()


def get_reports_for_patient(patient_id: int, db_path=None) -> list[dict]:
    """Get all reports for the patient in chronological order."""
    sb = _get_sb()
    result = (
        sb.table("reports")
        .select("*")
        .eq("patient_id", patient_id)
        .order("date", desc=False)
        .order("report_id", desc=False)
        .execute()
    )
    return result.data or []


def get_all_patients_with_reports() -> list[dict]:
    """Fetch all patients, newest first."""
    sb = _get_sb()
    return sb.table("patients").select("*").order("patient_id", desc=True).execute().data or []


def get_all_reports() -> list[dict]:
    """Fetch all reports, newest first."""
    sb = _get_sb()
    return sb.table("reports").select("*").order("report_id", desc=True).execute().data or []


# ── Progression ───────────────────────────────────────────────────────────────

def add_progression_record(
    patient_id: int,
    previous_severity: Optional[int],
    current_severity: int,
    status: str,
    date: Optional[str] = None,
    db_path=None,
) -> int:
    """Insert a progression record and return the new progression_id."""
    if date is None:
        date = datetime.date.today().isoformat()

    row = _insert_with_id_fallback(
        "progression",
        "progression_id",
        {
            "patient_id": patient_id,
            "previous_severity": previous_severity,
            "current_severity": current_severity,
            "status": status,
            "date": date,
        },
    )
    return int(row["progression_id"])


def get_progression_history(patient_id: int, db_path=None) -> list[dict]:
    """Get the progression history for a patient, oldest first."""
    sb = _get_sb()
    result = (
        sb.table("progression")
        .select("*")
        .eq("patient_id", patient_id)
        .order("date", desc=False)
        .order("progression_id", desc=False)
        .execute()
    )
    return result.data or []


# ── User authentication ───────────────────────────────────────────────────────

def count_users(db_path=None) -> int:
    """Return the number of registered users (used to bootstrap the first admin)."""
    sb = _get_sb()
    result = sb.table("user_roles").select("id", count="exact").execute()
    return result.count or 0


def create_user(username: str, password: str, email: str, role: str = "doctor", db_path=None) -> str:
    """
    Register a new user via Supabase Auth and store their role.
    Returns the Supabase user UUID as a string.
    """
    sb = _get_sb()

    # 1. Create the auth user
    auth_resp = sb.auth.admin.create_user(
        {
            "email": email.strip().lower(),
            "password": password,
            "email_confirm": True,   # auto-confirm; no email needed for internal tool
        }
    )
    user_id = auth_resp.user.id

    # 2. Store username + role in user_roles table
    sb.table("user_roles").insert(
        {
            "user_id": user_id,
            "username": username.strip(),
            "role": role,
        }
    ).execute()

    return user_id


def verify_user(email: str, password: str, db_path=None) -> Optional[dict]:
    """
    Authenticate via Supabase Auth.
    Returns a dict with user_id, username, role on success; None on failure.
    """
    sb_anon = create_client(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_KEY", ""),
    )
    try:
        auth_resp = sb_anon.auth.sign_in_with_password(
            {"email": email.strip().lower(), "password": password}
        )
    except Exception as exc:
        logger.warning("Supabase sign_in failed: %s", exc)
        return None

    if auth_resp.user is None:
        return None

    user_id = auth_resp.user.id

    # Fetch username + role from user_roles
    sb = _get_sb()
    role_row = (
        sb.table("user_roles")
        .select("username, role")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not role_row.data:
        return None

    return {
        "user_id": user_id,
        "username": role_row.data[0]["username"],
        "role": role_row.data[0]["role"],
        "email": email.strip().lower(),
    }


def get_user_by_username(username: str, db_path=None) -> Optional[dict]:
    """Fetch user info by username from user_roles table."""
    sb = _get_sb()
    result = (
        sb.table("user_roles")
        .select("user_id, username, role")
        .ilike("username", username.strip())
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ── SSO / Google OAuth helpers ────────────────────────────────────────────────

def get_oauth_sign_in_url(redirect_to: str, provider: str = "google") -> tuple[str, str]:
    """Generate Supabase OAuth sign-in URL and PKCE verifier."""
    sb = _get_sb()
    res = sb.auth.sign_in_with_oauth({
        "provider": provider,
        "options": {
            "redirect_to": redirect_to,
        },
    })
    verifier = sb.auth._storage.get_item(f"{sb.auth._storage_key}-code-verifier") or ""
    return res.url, verifier


def exchange_oauth_code(code: str, verifier: Optional[str] = None):
    """Exchange PKCE authorization code for authenticated user."""
    sb = _get_sb()
    auth_resp = sb.auth.exchange_code_for_session({
        "auth_code": code,
        "code_verifier": verifier or "",
    })
    return auth_resp.user


def get_user_from_access_token(access_token: str):
    """Retrieve user object given an OAuth access token."""
    sb = _get_sb()
    user_resp = sb.auth.get_user(access_token)
    return user_resp.user if user_resp else None


def get_or_create_sso_user(
    user_id: str,
    email: str,
    full_name: Optional[str] = None,
) -> dict:
    """
    Ensure an SSO authenticated user has an entry in user_roles.
    Returns a dict with user_id, username, role, email.
    """
    sb = _get_sb()
    role_row = (
        sb.table("user_roles")
        .select("user_id, username, role")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if role_row.data:
        return {
            "user_id": user_id,
            "username": role_row.data[0]["username"],
            "role": role_row.data[0]["role"],
            "email": email.strip().lower() if email else "",
        }

    # First user gets admin, subsequent get doctor
    role = "admin" if count_users() == 0 else "doctor"

    # Derive unique display username
    raw_name = (full_name or (email.split("@")[0] if email else "Doctor")).strip()
    cleaned_name = "".join(c for c in raw_name if c.isalnum() or c in (" ", "_", "-")).strip() or "Doctor"
    username = cleaned_name

    counter = 1
    while True:
        existing = (
            sb.table("user_roles")
            .select("id")
            .ilike("username", username)
            .limit(1)
            .execute()
        )
        if not existing.data:
            break
        counter += 1
        username = f"{cleaned_name} {counter}"

    sb.table("user_roles").insert({
        "user_id": user_id,
        "username": username,
        "role": role,
    }).execute()

    logger.info("Created new SSO user role entry: %s (%s, %s)", username, role, user_id)

    return {
        "user_id": user_id,
        "username": username,
        "role": role,
        "email": email.strip().lower() if email else "",
    }


# ── Supabase Storage helpers ──────────────────────────────────────────────────

BUCKET_IMAGES   = "retinal-images"
BUCKET_HEATMAPS = "heatmaps"
BUCKET_REPORTS  = "reports"

SIGNED_URL_EXPIRY = 3600  # 1 hour


def upload_file(local_path: str, bucket: str, remote_path: str) -> str:
    """
    Upload a local file to a Supabase Storage bucket.
    Returns the remote storage path (e.g. 'uploads/2024_scan.jpg').
    """
    sb = _get_sb()
    with open(local_path, "rb") as f:
        data = f.read()

    # Determine mime type
    ext = local_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "bmp": "image/bmp", "tiff": "image/tiff", "tif": "image/tiff",
        "pdf": "application/pdf",
    }
    content_type = mime_map.get(ext, "application/octet-stream")

    try:
        sb.storage.from_(bucket).upload(
            path=remote_path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        logger.info("Uploaded %s → %s/%s", local_path, bucket, remote_path)
    except Exception as exc:
        logger.warning("Storage upload failed (%s/%s): %s", bucket, remote_path, exc)

    return remote_path


def get_signed_url(bucket: str, remote_path: str, expiry: int = SIGNED_URL_EXPIRY) -> Optional[str]:
    """Generate a signed URL for a private Supabase Storage object."""
    if not remote_path:
        return None
    try:
        sb = _get_sb()
        result = sb.storage.from_(bucket).create_signed_url(remote_path, expiry)
        return result.get("signedURL") or result.get("signedUrl")
    except Exception as exc:
        logger.warning("Could not get signed URL for %s/%s: %s", bucket, remote_path, exc)
        return None


def storage_path_to_signed_url(storage_path: str) -> Optional[str]:
    """
    Convert a stored path like 'retinal-images/uploads/scan.jpg'
    into a signed URL. The bucket name is the first path segment.
    """
    if not storage_path or storage_path == "None":
        return None
    parts = storage_path.split("/", 1)
    if len(parts) == 2:
        bucket, remote = parts
        return get_signed_url(bucket, remote)
    return None


# ── Compatibility: add_report uses raw conn in app.py in one place ────────────
# app.py does:  conn.execute("UPDATE Reports SET llm_summary=? WHERE report_id=?")
# We patch that by making the "connection" object returned by get_connection()
# support that one UPDATE pattern through __getattr__ trickery.

class _SupabasePseudoConn:
    """Thin wrapper around Supabase client that mimics the sqlite3 Connection
    interface used in app.py for the LLM summary update and dashboard queries."""

    def __init__(self, sb: Client):
        self._sb = sb

    # ── context-manager support ──────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def close(self):
        pass   # no-op; Supabase client is stateless

    def commit(self):
        pass   # auto-committed by Supabase

    # ── execute shim ─────────────────────────────────────────────────────────
    def execute(self, sql: str, params: tuple = ()):
        """Handle the specific SQL patterns used directly in app.py."""
        sql_clean = sql.strip().upper()

        # UPDATE Reports SET llm_summary = ? WHERE report_id = ?
        if sql_clean.startswith("UPDATE REPORTS SET LLM_SUMMARY"):
            llm_summary, report_id = params
            self._sb.table("reports").update({"llm_summary": llm_summary}).eq("report_id", report_id).execute()
            return _FakeResult([])

        # SELECT * FROM Reports WHERE report_id = ?
        if sql_clean.startswith("SELECT * FROM REPORTS WHERE REPORT_ID"):
            report_id = params[0]
            result = self._sb.table("reports").select("*").eq("report_id", report_id).limit(1).execute()
            return _FakeResult(result.data or [])

        # SELECT * FROM PATIENTS WHERE patient_id = ?
        if sql_clean.startswith("SELECT * FROM PATIENTS WHERE PATIENT_ID"):
            patient_id = params[0]
            result = self._sb.table("patients").select("*").eq("patient_id", patient_id).limit(1).execute()
            return _FakeResult(result.data or [])

        # SELECT * FROM PROGRESSION WHERE PATIENT_ID = ? ORDER BY DATE ASC
        if "FROM PROGRESSION WHERE PATIENT_ID" in sql_clean:
            patient_id = params[0]
            result = (
                self._sb.table("progression")
                .select("*")
                .eq("patient_id", patient_id)
                .order("date", desc=False)
                .execute()
            )
            return _FakeResult(result.data or [])

        # SELECT * FROM PATIENTS ORDER BY PATIENT_ID DESC
        if "FROM PATIENTS ORDER BY PATIENT_ID DESC" in sql_clean:
            result = self._sb.table("patients").select("*").order("patient_id", desc=True).execute()
            return _FakeResult(result.data or [])

        # SELECT * FROM REPORTS WHERE PATIENT_ID = ? ORDER BY DATE DESC
        if "FROM REPORTS WHERE PATIENT_ID" in sql_clean and "ORDER BY DATE DESC" in sql_clean:
            patient_id = params[0]
            result = (
                self._sb.table("reports")
                .select("*")
                .eq("patient_id", patient_id)
                .order("date", desc=True)
                .execute()
            )
            return _FakeResult(result.data or [])

        # SELECT * FROM PROGRESSION WHERE PATIENT_ID = ? ORDER BY DATE DESC
        if "FROM PROGRESSION WHERE PATIENT_ID" in sql_clean and "ORDER BY DATE DESC" in sql_clean:
            patient_id = params[0]
            result = (
                self._sb.table("progression")
                .select("*")
                .eq("patient_id", patient_id)
                .order("date", desc=True)
                .execute()
            )
            return _FakeResult(result.data or [])

        logger.warning("Unhandled SQL in pseudo-conn: %s", sql[:120])
        return _FakeResult([])


class _FakeResult:
    """Mimic sqlite3.Cursor with fetchone() / fetchall()."""
    def __init__(self, data: list):
        self._data = data

    def fetchone(self):
        return self._data[0] if self._data else None

    def fetchall(self):
        return self._data


def get_connection(db_path=None):
    """Return a pseudo-connection that routes SQL to Supabase."""
    return _SupabasePseudoConn(_get_sb())


# Re-export so progression.py import still works
__all__ = [
    "DB_PATH", "get_connection", "init_db",
    "add_patient", "get_patient",
    "add_report", "get_report", "update_report_llm_summary",
    "get_reports_for_patient",
    "add_progression_record", "get_progression_history",
    "count_users", "create_user", "verify_user", "get_user_by_username",
    "upload_file", "get_signed_url", "storage_path_to_signed_url",
    "BUCKET_IMAGES", "BUCKET_HEATMAPS", "BUCKET_REPORTS",
]
