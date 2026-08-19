from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

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


def add_patient(name: str, age: int, gender: str, diabetes_history: str, db_path: str | Path = DB_PATH) -> int:
    """Add a patient to the Patients table and return the new patient_id."""
    conn = get_connection(db_path)
    try:
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
        date = date.today().isoformat()

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
        date = date.today().isoformat()

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
