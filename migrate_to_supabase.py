"""
migrate_to_supabase.py  —  Run ONCE after Supabase tables are created.
Usage:
    .venv/bin/python migrate_to_supabase.py
"""
from __future__ import annotations
import os, sqlite3, sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SQLITE_DB = Path(__file__).resolve().parent / "database.db"

def get_sb() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or "YOUR_PROJECT_ID" in url:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env first."); sys.exit(1)
    return create_client(url, key)

def migrate():
    if not SQLITE_DB.exists():
        print(f"SQLite DB not found at {SQLITE_DB}"); return
    print(f"Reading from {SQLITE_DB}")
    conn = sqlite3.connect(str(SQLITE_DB)); conn.row_factory = sqlite3.Row
    sb = get_sb()

    # Patients
    patients = conn.execute("SELECT * FROM Patients").fetchall()
    print(f"\nMigrating {len(patients)} patient(s)...")
    for p in patients:
        try:
            sb.table("patients").upsert({"patient_id": p["patient_id"], "name": p["name"],
                "age": p["age"], "gender": p["gender"], "diabetes_history": p["diabetes_history"]},
                on_conflict="patient_id").execute()
            print(f"  ✓ Patient {p['patient_id']}: {p['name']}")
        except Exception as e:
            print(f"  ✗ Patient {p['patient_id']}: {e}")

    # Reports
    reports = conn.execute("SELECT * FROM Reports").fetchall()
    print(f"\nMigrating {len(reports)} report(s)...")
    for r in reports:
        try:
            sb.table("reports").upsert({
                "report_id": r["report_id"], "patient_id": r["patient_id"],
                "image_path": r["image_path"], "severity": r["severity"],
                "confidence": r["confidence"], "heatmap_path": r["heatmap_path"],
                "recommendation": r["recommendation"], "llm_summary": r["llm_summary"],
                "date": r["date"]}, on_conflict="report_id").execute()
            print(f"  ✓ Report {r['report_id']} (patient {r['patient_id']})")
        except Exception as e:
            print(f"  ✗ Report {r['report_id']}: {e}")

    # Progression
    prog = conn.execute("SELECT * FROM Progression").fetchall()
    print(f"\nMigrating {len(prog)} progression record(s)...")
    for p in prog:
        try:
            sb.table("progression").upsert({
                "progression_id": p["progression_id"], "patient_id": p["patient_id"],
                "previous_severity": p["previous_severity"], "current_severity": p["current_severity"],
                "status": p["status"], "date": p["date"]},
                on_conflict="progression_id").execute()
            print(f"  ✓ Progression {p['progression_id']} — {p['status']}")
        except Exception as e:
            print(f"  ✗ Progression {p['progression_id']}: {e}")

    try:
        users = conn.execute("SELECT * FROM Users").fetchall()
        print(f"\nFound {len(users)} SQLite user(s) — passwords CANNOT be migrated.")
        print("Re-register all users at /register after starting the app:")
        for u in users: print(f"  - {u['username']} ({u['role']})")
    except: pass

    conn.close()
    print("\nDone! Now restart Flask and register your users via /register")

if __name__ == "__main__":
    migrate()
