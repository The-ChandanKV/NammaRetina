"""
NammaRetina - PDF Report Generator (Phase 12)
Generates a professional medical report PDF using ReportLab.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)

logger = logging.getLogger(__name__)

PAGE_WIDTH, PAGE_HEIGHT = A4


def _get_styles():
    """Build custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a5276"),
        spaceAfter=6,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#5d6d7e"),
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1a5276"),
        spaceBefore=16,
        spaceAfter=6,
        borderWidth=0,
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "SmallGray",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#7f8c8d"),
        alignment=TA_CENTER,
    ))

    return styles


def generate_pdf(
    report_id: int,
    patient: dict,
    severity: int,
    severity_label: str,
    confidence: float,
    recommendation: str,
    llm_summary: str,
    heatmap_path: str | None = None,
    progression_graph_path: str | None = None,
    simulation_path: str | None = None,
    report_date: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Generate a PDF medical report and return the file path.

    Parameters
    ----------
    report_id : int
        The database report ID.
    patient : dict
        Patient record with keys: name, age, gender, diabetes_history.
    severity : int
        Predicted DR severity (0-4).
    severity_label : str
        Human-readable severity label.
    confidence : float
        Model confidence (0.0-1.0).
    recommendation : str
        Treatment recommendation string.
    llm_summary : str
        LLM-generated plain-language explanation.
    heatmap_path : str, optional
        Path to the Grad-CAM heatmap image.
    progression_graph_path : str, optional
        Path to the progression graph image.
    simulation_path : str, optional
        Path to the disease spread simulation image.
    report_date : str, optional
        Report date string (defaults to today).
    output_dir : str, optional
        Directory to save the PDF (defaults to reports/).

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "reports")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"report_{report_id}.pdf"

    styles = _get_styles()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    elements = []

    # ── Title ────────────────────────────────────────────────────────────────
    elements.append(Paragraph("NammaRetina", styles["ReportTitle"]))
    elements.append(Paragraph(
        "AI-Based Diabetic Retinopathy Diagnostic Report",
        styles["ReportSubtitle"],
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=colors.HexColor("#1a5276"), spaceAfter=12,
    ))

    # ── Patient Details ──────────────────────────────────────────────────────
    elements.append(Paragraph("Patient Information", styles["SectionHeading"]))

    patient_data = [
        ["Name", str(patient.get("name", "N/A"))],
        ["Age", str(patient.get("age", "N/A"))],
        ["Gender", str(patient.get("gender", "N/A"))],
        ["Diabetes History", str(patient.get("diabetes_history", "N/A"))],
        ["Report Date", report_date],
        ["Report ID", str(report_id)],
    ]
    patient_table = Table(patient_data, colWidths=[120, 350])
    patient_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a5276")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 10))

    # ── Prediction Result ────────────────────────────────────────────────────
    elements.append(Paragraph("Diagnosis Result", styles["SectionHeading"]))

    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)

    # Severity colour coding
    severity_colours = {
        0: colors.HexColor("#27ae60"),
        1: colors.HexColor("#f39c12"),
        2: colors.HexColor("#e67e22"),
        3: colors.HexColor("#e74c3c"),
        4: colors.HexColor("#c0392b"),
    }
    sev_color = severity_colours.get(severity, colors.black)

    prediction_data = [
        ["Severity Level", f"{severity_label} (Stage {severity}/4)"],
        ["Confidence", f"{confidence_pct}%"],
    ]
    prediction_table = Table(prediction_data, colWidths=[120, 350])
    prediction_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (1, 0), (1, 0), sev_color),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(prediction_table)
    elements.append(Spacer(1, 10))

    # ── Grad-CAM Heatmap ─────────────────────────────────────────────────────
    if heatmap_path and Path(heatmap_path).exists():
        elements.append(Paragraph("Grad-CAM Heatmap", styles["SectionHeading"]))
        elements.append(Paragraph(
            "Highlighted regions indicate areas the AI focused on for its prediction.",
            styles["BodyText2"],
        ))
        try:
            img = RLImage(heatmap_path, width=3.5 * inch, height=3.5 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception as exc:
            logger.warning("Could not embed heatmap image: %s", exc)
        elements.append(Spacer(1, 10))

    # ── Progression Graph ────────────────────────────────────────────────────
    if progression_graph_path and Path(progression_graph_path).exists():
        elements.append(Paragraph("Progression History", styles["SectionHeading"]))
        elements.append(Paragraph(
            "Severity trend across all recorded visits for this patient.",
            styles["BodyText2"],
        ))
        try:
            img = RLImage(progression_graph_path, width=4.5 * inch, height=2.8 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception as exc:
            logger.warning("Could not embed progression graph: %s", exc)
        elements.append(Spacer(1, 10))

    # ── Simulation ───────────────────────────────────────────────────────────
    if simulation_path and Path(simulation_path).exists():
        elements.append(Paragraph("Disease Spread Simulation", styles["SectionHeading"]))
        elements.append(Paragraph(
            "Probabilistic visualisation of possible future retinal changes. "
            "This is not a clinically validated forecast.",
            styles["BodyText2"],
        ))
        try:
            img = RLImage(simulation_path, width=3.5 * inch, height=3.5 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        except Exception as exc:
            logger.warning("Could not embed simulation image: %s", exc)
        elements.append(Spacer(1, 10))

    # ── Treatment Recommendation ─────────────────────────────────────────────
    elements.append(Paragraph("Treatment Recommendation", styles["SectionHeading"]))
    elements.append(Paragraph(recommendation, styles["BodyText2"]))
    elements.append(Spacer(1, 6))

    # ── LLM Explanation ──────────────────────────────────────────────────────
    elements.append(Paragraph("AI-Generated Explanation", styles["SectionHeading"]))
    elements.append(Paragraph(llm_summary, styles["BodyText2"]))
    elements.append(Spacer(1, 20))

    # ── Footer ───────────────────────────────────────────────────────────────
    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#bdc3c7"), spaceAfter=6,
    ))
    elements.append(Paragraph(
        "This report was generated by NammaRetina — AI-Based Multi-Agent "
        "Diabetic Retinopathy Diagnostic and Progression Analysis System. "
        "This is an AI-assisted analysis and should be reviewed by a qualified "
        "medical professional before any clinical decisions are made.",
        styles["SmallGray"],
    ))

    # Build PDF
    doc.build(elements)
    logger.info("PDF report generated: %s", pdf_path)
    return str(pdf_path)
