"""Longitudinal diabetic retinopathy progression analysis.

Distinguishes observed historical severity changes (from recorded examinations)
from illustrative future spatial simulations. Does not produce clinical forecasts
or invented risk percentages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

DR_CLASSES = {
    0: "No apparent DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def _ordinal_delta_label(delta: int) -> str:
    if delta > 0:
        return "Increased"
    if delta < 0:
        return "Decreased"
    return "Unchanged"


def _contextual_risk_notes(patient: Optional[dict]) -> dict:
    """Surface only risk-factor text that already exists on the patient record.

    The schema stores a free-text ``diabetes_history`` field. Structured HbA1c,
    blood pressure, and diabetes type are not present — they are not invented.
    """
    history = ""
    if patient:
        history = str(patient.get("diabetes_history") or "").strip()
    notes = []
    if history and history.lower() not in {"not provided", "none", "n/a", "na", "-"}:
        notes.append(
            {
                "factor": "Recorded diabetes history",
                "value": history,
                "evidence_level": "descriptive",
                "note": "Free-text clinical note only; not a validated risk score.",
            }
        )
    return {
        "available_factors": notes,
        "unavailable_structured_fields": [
            "diabetes_duration",
            "diabetes_type",
            "HbA1c",
            "blood_pressure",
        ],
        "scoring_method": None,
        "disclaimer": (
            "No validated progression-risk percentage is computed. "
            "Only documented patient fields are shown."
        ),
    }


def analyse_longitudinal_progression(
    reports: list[dict],
    patient: Optional[dict] = None,
) -> dict:
    """Describe chronological severity changes across recorded examinations.

    Severity grades are ordinal model/clinical labels (0–4). Differences between
    grades are not treated as continuous biological measurements, and apparent
    improvement may reflect misclassification rather than recovery.
    """
    usable = [
        dict(row)
        for row in (reports or [])
        if float(row.get("confidence") or 0) > 0 and row.get("severity") is not None
    ]
    usable.sort(
        key=lambda row: (
            _parse_date(row.get("date")) or datetime.min,
            int(row.get("report_id") or 0),
        )
    )

    examinations = []
    for row in usable:
        severity = int(row["severity"])
        examinations.append(
            {
                "report_id": int(row.get("report_id") or 0),
                "date": str(row.get("date") or ""),
                "severity": severity,
                "severity_label": DR_CLASSES.get(severity, "Unknown"),
                "confidence": float(row.get("confidence") or 0),
                "image_path": row.get("image_path") or "",
                "heatmap_path": row.get("heatmap_path") or "",
            }
        )

    transitions = []
    for previous, current in zip(examinations, examinations[1:]):
        prev_date = _parse_date(previous["date"])
        curr_date = _parse_date(current["date"])
        interval_days = None
        if prev_date and curr_date:
            interval_days = (curr_date - prev_date).days
        delta = current["severity"] - previous["severity"]
        transitions.append(
            {
                "from_report_id": previous["report_id"],
                "to_report_id": current["report_id"],
                "from_date": previous["date"],
                "to_date": current["date"],
                "from_severity": previous["severity"],
                "to_severity": current["severity"],
                "from_label": previous["severity_label"],
                "to_label": current["severity_label"],
                "ordinal_delta": delta,
                "change": _ordinal_delta_label(delta),
                "interval_days": interval_days,
            }
        )

    overall = "Insufficient history"
    if len(examinations) == 1:
        overall = "Single examination — no longitudinal comparison"
    elif transitions:
        last = transitions[-1]["change"]
        if all(t["change"] == "Unchanged" for t in transitions):
            overall = "Recorded severity unchanged across examinations"
        elif all(t["ordinal_delta"] >= 0 for t in transitions) and any(t["ordinal_delta"] > 0 for t in transitions):
            overall = "Recorded severity increased over follow-up"
        elif all(t["ordinal_delta"] <= 0 for t in transitions) and any(t["ordinal_delta"] < 0 for t in transitions):
            overall = "Recorded severity decreased over follow-up"
        else:
            overall = f"Mixed ordinal changes; most recent interval: {last}"

    first = examinations[0] if examinations else None
    last = examinations[-1] if examinations else None
    span_days = None
    if first and last:
        d0, d1 = _parse_date(first["date"]), _parse_date(last["date"])
        if d0 and d1:
            span_days = (d1 - d0).days

    summary_parts = []
    n = len(examinations)
    summary_parts.append(f"{n} examination{'s' if n != 1 else ''} with valid model predictions.")
    if first and last and n >= 2:
        summary_parts.append(
            f"Recorded severity moved from {first['severity_label']} ({first['date']}) "
            f"to {last['severity_label']} ({last['date']})."
        )
        if span_days is not None:
            summary_parts.append(f"Observation span: {span_days} days.")
    summary_parts.append(overall + ".")
    summary_parts.append(
        "Ordinal grade changes are descriptive of recorded labels only and are not "
        "proof of biological progression or recovery; model misclassification can "
        "create apparent change."
    )

    return {
        "n_examinations": n,
        "examinations": examinations,
        "transitions": transitions,
        "overall_trend": overall,
        "observation_span_days": span_days,
        "first_exam": first,
        "latest_exam": last,
        "previous_vs_current": {
            "previous": examinations[-2] if n >= 2 else None,
            "current": last,
        },
        "summary": " ".join(summary_parts),
        "risk_context": _contextual_risk_notes(patient),
        "forecasting": {
            "available": False,
            "reason": (
                "No patient-level longitudinal training set with timed outcomes is "
                "available in this repository. Supervised progression-risk modeling "
                "is therefore not implemented."
            ),
            "future_requirements": [
                "Patient identifiers linking serial examinations",
                "Confirmed clinical grades at defined follow-up horizons",
                "Censoring indicators for incomplete follow-up",
                "Patient-level train/test separation and external temporal validation",
                "Calibrated risk estimates with a stated prediction horizon",
            ],
        },
        "limitations": [
            "Severity values may be model predictions rather than adjudicated clinical grades.",
            "Differences between ordinal stages are not continuous measures of disease burden.",
            "Apparent improvement does not establish clinical recovery.",
            "Illustrative spatial simulations are separate from this historical analysis.",
        ],
        "analysis_type": "descriptive_longitudinal",
    }
