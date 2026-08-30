from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from database import DB_PATH, add_progression_record, get_patient, get_reports_for_patient


def _validate_severity(severity: int) -> int:
    """Validate the DR severity level used in progression tracking."""
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError("Severity must be an integer from 0 to 4.")
    if severity < 0 or severity > 4:
        raise ValueError("Severity must be an integer from 0 to 4.")
    return severity


def track_progression(patient_id: int, current_severity: int, visit_date: str | None = None, db_path=DB_PATH):
    """Track the change in pathology severity between the most recent prior report and the current one.

    ``visit_date`` stamps the progression record with the report's date (so the
    progression timeline matches the report timeline); it defaults to today only
    when a caller does not supply one.
    """
    if get_patient(patient_id, db_path=db_path) is None:
        raise ValueError(f"Patient {patient_id} does not exist.")

    current_severity = _validate_severity(current_severity)
    reports = get_reports_for_patient(patient_id, db_path=db_path)

    previous_severity = None
    if reports:
        last_recorded_severity = int(reports[-1]["severity"])
        if last_recorded_severity == current_severity and len(reports) > 1:
            previous_severity = int(reports[-2]["severity"])
        elif last_recorded_severity == current_severity and len(reports) == 1:
            previous_severity = None
        else:
            previous_severity = last_recorded_severity

    if previous_severity is None:
        status = "Initial"
    elif current_severity > previous_severity:
        status = "Worsened"
    elif current_severity < previous_severity:
        status = "Improved"
    else:
        status = "Stable"

    if visit_date is None:
        visit_date = datetime.now().strftime("%Y-%m-%d")

    record = {
        "patient_id": patient_id,
        "previous_severity": previous_severity,
        "current_severity": current_severity,
        "status": status,
        "date": visit_date,
    }
    add_progression_record(
        patient_id=patient_id,
        previous_severity=previous_severity,
        current_severity=current_severity,
        status=status,
        date=visit_date,
        db_path=db_path,
    )
    return record


def plot_progression(patient_id: int, db_path=DB_PATH) -> str:
    """Save a severity-over-time graph for the patient to the reports directory."""
    if get_patient(patient_id, db_path=db_path) is None:
        raise ValueError(f"Patient {patient_id} does not exist.")

    reports = get_reports_for_patient(patient_id, db_path=db_path)
    if not reports:
        raise ValueError(f"No reports found for patient {patient_id}.")

    ordered_reports = sorted(
        reports,
        key=lambda row: datetime.strptime(str(row["date"]), "%Y-%m-%d") if row["date"] else datetime.min,
    )

    dates = [str(row["date"]) for row in ordered_reports]
    severities = [int(row["severity"]) for row in ordered_reports]

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f"progression_{patient_id}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dates, severities, marker="o", linewidth=2, color="tab:red")
    ax.set_ylim(0, 4)
    ax.set_yticks(range(5))
    ax.set_ylabel("DR Severity")
    ax.set_xlabel("Date")
    ax.set_title(f"Patient {patient_id} progression")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return str(output_path)
