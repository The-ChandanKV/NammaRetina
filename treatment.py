from __future__ import annotations


def get_treatment_recommendation(severity: int) -> str:
    """Return a simple rule-based treatment message for a given DR severity."""
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError("Severity must be an integer from 0 to 4.")
    if severity < 0 or severity > 4:
        raise ValueError("Severity must be an integer from 0 to 4.")

    mapping = {
        0: "Routine yearly screening",
        1: "Monitor blood sugar and follow-up",
        2: "Ophthalmologist consultation advised",
        3: "Immediate specialist consultation",
        4: "Urgent retinal treatment required",
    }
    return mapping[severity]
