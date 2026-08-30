from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = Path(__file__).resolve().parent / "database.db"


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection with foreign key enforcement enabled."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = DB_PATH) -> str:
    """Initialize the project database and create required tables."""
    conn = get_connection(db_path)
    try:
        create_tables(conn)
        conn.commit()
        return str(Path(db_path))
    finally:
        conn.close()


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the Patients, Reports, and Progression tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Patients (
            patient_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            diabetes_history TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Reports (
            report_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            image_path TEXT,
            severity INTEGER,
            confidence FLOAT,
            heatmap_path TEXT,
            recommendation TEXT,
            llm_summary TEXT,
            date DATE,
            FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Progression (
            progression_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            previous_severity INTEGER,
            current_severity INTEGER,
            status TEXT,
            date DATE,
            FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'doctor',
            created_at TEXT
        )
        """
    )


def add_patient(name: str, age: int, gender: str, diabetes_history: str, db_path: str | Path = DB_PATH) -> int:
    """Return the patient_id for this patient, creating the record only if a
    matching patient (same name/age/gender, case-insensitive) does not exist.

    Deduping here is what makes multi-visit progression work: re-uploading a scan
    for the same person reuses their patient_id instead of creating a new patient.
    """
    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            """
            SELECT patient_id FROM Patients
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
              AND age = ?
              AND LOWER(TRIM(gender)) = LOWER(TRIM(?))
            ORDER BY patient_id ASC
            LIMIT 1
            """,
            (name, age, gender),
        ).fetchone()
        if existing is not None:
            return int(existing["patient_id"])

        cursor = conn.execute(
            """
            INSERT INTO Patients (name, age, gender, diabetes_history)
            VALUES (?, ?, ?, ?)
            """,
            (name, age, gender, diabetes_history),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_patient(patient_id: int, db_path: str | Path = DB_PATH) -> Optional[sqlite3.Row]:
    """Fetch a single patient by patient_id."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM Patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        return row
    finally:
        conn.close()


def add_report(
    patient_id: int,
    image_path: str,
    severity: int,
    confidence: float,
    heatmap_path: str,
    recommendation: str,
    llm_summary: str,
    date: Optional[str] = None,
    db_path: str | Path = DB_PATH,
) -> int:
    """Add a report to the Reports table and return the new report_id."""
    if date is None:
        date = datetime.date.today().isoformat()

    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO Reports (
                patient_id, image_path, severity, confidence, heatmap_path,
                recommendation, llm_summary, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_id, image_path, severity, confidence, heatmap_path, recommendation, llm_summary, date),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_reports_for_patient(patient_id: int, db_path: str | Path = DB_PATH) -> list[sqlite3.Row]:
    """Get all reports for the given patient in chronological order."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM Reports WHERE patient_id = ? ORDER BY date ASC, report_id ASC",
            (patient_id,),
        ).fetchall()
        return list(rows)
    finally:
        conn.close()


def add_progression_record(
    patient_id: int,
    previous_severity: Optional[int],
    current_severity: int,
    status: str,
    date: Optional[str] = None,
    db_path: str | Path = DB_PATH,
) -> int:
    """Store an individual progression status for a patient."""
    if date is None:
        date = datetime.date.today().isoformat()

    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO Progression (patient_id, previous_severity, current_severity, status, date)
            VALUES (?, ?, ?, ?, ?)
            """,
            (patient_id, previous_severity, current_severity, status, date),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_progression_history(patient_id: int, db_path: str | Path = DB_PATH) -> list[sqlite3.Row]:
    """Get the progression history for the given patient."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM Progression WHERE patient_id = ? ORDER BY date ASC, progression_id ASC",
            (patient_id,),
        ).fetchall()
        return list(rows)
    finally:
        conn.close()


# ─── User authentication ────────────────────────────────────────────────────

def count_users(db_path: str | Path = DB_PATH) -> int:
    """Return the number of registered users (used to bootstrap the first admin)."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM Users").fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()


def get_user_by_username(username: str, db_path: str | Path = DB_PATH) -> Optional[sqlite3.Row]:
    """Fetch a user row by username, or None if not found."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT * FROM Users WHERE username = ?", (username.strip(),)
        ).fetchone()
    finally:
        conn.close()


def create_user(username: str, password: str, role: str = "doctor", db_path: str | Path = DB_PATH) -> int:
    """Create a user with a werkzeug-hashed password and return the new user_id.

    Raises sqlite3.IntegrityError if the username already exists (UNIQUE constraint).
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO Users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (
                username.strip(),
                generate_password_hash(password),
                role,
                datetime.datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def verify_user(username: str, password: str, db_path: str | Path = DB_PATH) -> Optional[sqlite3.Row]:
    """Return the user Row if the username exists and the password matches, else None."""
    user = get_user_by_username(username, db_path=db_path)
    if user is None:
        return None
    if check_password_hash(user["password_hash"], password):
        return user
    return None
