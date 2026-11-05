"""One report contract shared by the patient UI and PDF renderer.

No model, LLM, database or storage calls belong here. A report is a snapshot of
the selected examination, not of the patient's most recent visit.
"""

from __future__ import annotations

import math

from config import DR_CLASSES
from progression_analysis import analyse_longitudinal_progression
from treatment import get_treatment_recommendation

REPORT_FORMAT_VERSION = 2
REPORT_SECTIONS = (
    ("patient", "Patient information"),
    ("screening", "Screening result"),
    ("interpretation", "Standardized interpretation"),
    ("recommendation", "Recommendation"),
    ("history", "History through this examination"),
    ("imaging", "Scan and Grad-CAM"),
    ("simulation", "Illustrative spatial simulation"),
    ("supplemental", "Supplementary AI explanation"),
)
REPORT_DISCLAIMER = (
    "AI-assisted screening support; not a confirmed diagnosis. Model confidence "
    "is not the probability that a diagnosis is correct. Grade changes describe "
    "recorded predictions and do not prove biological progression or recovery. "
    "A qualified clinician must review findings before any care decision."
)


def normalize_report(report: dict) -> dict:
    """Coerce stored numeric fields and reject unavailable/invalid predictions.

    Older imports may store a percentage rather than the current 0–1 fraction.
    Normalize it once so every view displays the same score.
    """
    row = dict(report)
    try:
        confidence = float(row.get("confidence") or 0)
        if 1 < confidence <= 100:
            confidence /= 100
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            confidence = 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        raw_severity = float(row.get("severity"))
        severity = int(raw_severity)
        if severity not in DR_CLASSES or severity != raw_severity:
            severity = None
    except (TypeError, ValueError, OverflowError):
        severity = None
    available = severity is not None and confidence > 0
    row.update(
        severity=severity,
        confidence=confidence if available else 0.0,
        model_used=available,
        confidence_pct=round(confidence * 100, 1) if available else None,
        severity_label=DR_CLASSES[severity] if available else "Analysis unavailable",
    )
    return row


def report_order(report: dict) -> tuple[str, int]:
    return str(report.get("date") or "")[:10], int(report.get("report_id") or 0)


def pdf_filename(report_id: int) -> str:
    """Version caches so older PDFs cannot bypass the standardized renderer."""
    return f"report_{int(report_id)}_v{REPORT_FORMAT_VERSION}.pdf"


def build_report_context(report: dict, patient: dict, reports: list[dict]) -> dict:
    current = normalize_report(report)
    # Filter by patient as well as date/ID; never include a subsequent visit.
    history = [normalize_report(row) for row in reports or []
               if row.get("patient_id") in (None, current.get("patient_id"))
               and row.get("report_id") != current.get("report_id")
               and report_order(row) <= report_order(current)]
    history.append(current)
    history.sort(key=report_order)
    longitudinal = analyse_longitudinal_progression(history, patient=dict(patient))
    examinations = longitudinal["examinations"]
    prior = [row for row in examinations if row["report_id"] != current["report_id"]]
    previous = prior[-1] if prior else None
    status = "Unavailable"
    if current["model_used"]:
        status = "Initial"
        if previous:
            delta = current["severity"] - previous["severity"]
            status = "Worsened" if delta > 0 else "Improved" if delta < 0 else "Stable"

    progression = []
    for idx, row in enumerate(examinations):
        prev = examinations[idx - 1] if idx else None
        delta = row["severity"] - prev["severity"] if prev else 0
        progression.append({
            "report_id": row["report_id"], "date": row["date"],
            "previous_severity": prev["severity"] if prev else None,
            "current_severity": row["severity"],
            "status": ("Initial" if prev is None else
                       "Worsened" if delta > 0 else "Improved" if delta < 0 else "Stable"),
        })

    if not current["model_used"]:
        classification = "No valid model prediction is stored for this examination."
        score = "Model confidence is unavailable."
        comparison = "This record cannot be used to assess a change in severity."
        recommendation = "A new scan analysis with a loaded model is required."
    else:
        classification = (f"The model classified this examination as {current['severity_label']} "
                          f"(grade {current['severity']}/4). This is a screening prediction.")
        score = (f"The recorded model confidence is {current['confidence_pct']:.1f}%. "
                 "This score does not establish diagnostic certainty or overall model accuracy.")
        if previous:
            comparison = (f"Compared with report #{previous['report_id']} ({previous['date']}), "
                          f"the recorded grade changed from {previous['severity']}/4 to "
                          f"{current['severity']}/4. Recorded status: {status}.")
        else:
            comparison = "This is the first valid examination on record; no prior comparison is available."
        recommendation = str(current.get("recommendation") or "").strip()
        if recommendation.lower() in {"", "none", "null", "pending", "(pending)", "placeholder"}:
            recommendation = get_treatment_recommendation(current["severity"])

    summary = str(current.get("llm_summary") or "").strip()
    if not current["model_used"] or summary.lower() in {
        "", "none", "null", "pending", "(pending)", "placeholder",
    }:
        summary = ""
    return {
        "report": current, "patient": dict(patient), "history": history,
        "longitudinal": longitudinal, "progression": progression,
        "progression_status": status, "recommendation": recommendation,
        "interpretation": [
            {"label": "Classification", "text": classification},
            {"label": "Model confidence", "text": score},
            {"label": "Previous examination", "text": comparison},
        ],
        "llm_summary": summary, "sections": REPORT_SECTIONS,
        "format_version": REPORT_FORMAT_VERSION, "disclaimer": REPORT_DISCLAIMER,
    }
