"""Standardized NammaRetina screening PDFs, using the same data as the web UI."""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

from reporting import build_report_context, pdf_filename

logger = logging.getLogger(__name__)


def generate_pdf(
    report_id: int, patient: dict, severity: int, severity_label: str,
    confidence: float, recommendation: str, llm_summary: str,
    heatmap_path: str | None = None, progression_graph_path: str | None = None,
    simulation_path: str | None = None, report_date: str | None = None,
    output_dir: str | None = None, report_context: dict | None = None,
    image_path: str | None = None,
) -> str:
    """Render all eight report sections, including explicit missing-data states.

    Dynamic text is plain text, never ReportLab markup. Legacy callers remain
    supported; web callers supply the exact examination context shown on screen.
    Patient-wide progression graphs are intentionally not used in visit reports.
    """
    context = report_context or build_report_context({
        "report_id": report_id, "patient_id": patient.get("patient_id"),
        "severity": severity, "confidence": confidence,
        "recommendation": recommendation, "llm_summary": llm_summary,
        "date": report_date or datetime.now().strftime("%Y-%m-%d"),
    }, patient, [])
    report = context["report"]
    patient = context["patient"]
    out_dir = Path(output_dir or Path(__file__).resolve().parent / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / pdf_filename(report_id)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("NRTitle", parent=styles["Title"], fontSize=23,
                              textColor=colors.HexColor("#008b80"), alignment=TA_CENTER))
    styles.add(ParagraphStyle("NRBody", parent=styles["BodyText"], fontSize=9,
                              leading=13, spaceAfter=7, splitLongWords=True))
    styles.add(ParagraphStyle("NRSmall", parent=styles["NRBody"], fontSize=8, leading=11,
                              textColor=colors.HexColor("#526476")))
    styles.add(ParagraphStyle("NRHeading", parent=styles["Heading2"], fontSize=12,
                              textColor=colors.HexColor("#008b80"), spaceBefore=13,
                              spaceAfter=7, keepWithNext=True))

    def text(value, small=False):
        value = "Not provided" if value is None or str(value).strip() == "" else str(value)
        return Paragraph(escape(value).replace("\n", "<br/>"), styles["NRSmall" if small else "NRBody"])

    width = A4[0] - 40 * mm
    elements = [Paragraph("NammaRetina", styles["NRTitle"]),
                text("AI-assisted diabetic retinopathy screening report"),
                text(f"Report #{report_id} | Examination: {report['date']} | Format v{context['format_version']}", True)]

    def heading(section_id):
        idx, title = next((i, title) for i, (key, title) in enumerate(context["sections"], 1) if key == section_id)
        elements.append(Paragraph(f"{idx:02d}. {escape(title)}", styles["NRHeading"]))

    def data_table(rows, header=False):
        cells = [[text(v, True) for v in row] for row in rows]
        columns = len(rows[0])
        table = Table(cells, colWidths=([38 * mm, width - 38 * mm] if columns == 2 else [width / columns] * columns),
                      hAlign="LEFT", repeatRows=1 if header else 0, splitInRow=1)
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0 if not header else -1, -1 if not header else 0), colors.HexColor("#eef7f6")),
            ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#dfe7ef")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    def image(path, label):
        elements.append(text(label))
        if path and Path(path).is_file():
            try:
                # Decode now: corrupt image files must not abort doc.build later.
                with Image.open(path) as source:
                    source.load()
                    w, h = source.size
                scale = min(width / w, 60 * mm / h)
                flow = RLImage(str(path), width=w * scale, height=h * scale)
                flow.hAlign = "CENTER"
                elements.extend([flow, Spacer(1, 3 * mm)])
                return
            except Exception:
                logger.warning("Could not embed %s", label)
        elements.append(text("Image unavailable for this examination.", True))

    heading("patient")
    data_table([(label, patient.get(key)) for label, key in [
        ("Patient ID", "patient_id"), ("Name", "name"), ("Age", "age"),
        ("Gender", "gender"), ("Diabetes history", "diabetes_history"),
    ]])
    heading("screening")
    data_table([
        ("Classification", report["severity_label"]),
        ("Grade", f"{report['severity']}/4" if report["model_used"] else "Unavailable"),
        ("Model confidence", f"{report['confidence_pct']:.1f}%" if report["model_used"] else "Unavailable"),
        ("Recorded change", context["progression_status"]),
    ])
    elements.append(text("Model confidence is separate from the model's measured accuracy.", True))
    heading("interpretation")
    for item in context["interpretation"]:
        elements.append(text(item["label"] + ": " + item["text"]))
    heading("recommendation")
    elements.append(text(context["recommendation"]))
    elements.append(text("The responsible clinician should confirm the appropriate next step.", True))
    heading("history")
    elements.append(text(context["longitudinal"]["summary"]))
    if context["progression"]:
        data_table([("Report", "Date", "Grade", "Recorded change")] + [
            (f"#{row['report_id']}", row["date"], f"{row['current_severity']}/4", row["status"])
            for row in context["progression"]
        ], header=True)
    else:
        elements.append(text("No valid model predictions are available for comparison."))
    elements.append(text("Only examinations through this report are included. This is an observed history, not a clinical forecast.", True))
    heading("imaging")
    image(image_path, "Original retinal scan")
    image(heatmap_path, "Grad-CAM heatmap")
    elements.append(text("Grad-CAM highlights model attention; it does not independently confirm retinal lesions.", True))
    heading("simulation")
    image(simulation_path, "Illustrative spatial simulation - not a clinical forecast")
    elements.append(text("Deterministic Grad-CAM attention remapping. This does not predict future lesions or disease spread.", True))
    heading("supplemental")
    elements.append(text(context["llm_summary"] or "No supplementary AI explanation is stored for this examination."))
    elements.append(text("Supplementary AI text is stored narrative and requires clinician review. The standardized interpretation above is based on recorded examination fields.", True))
    elements.append(text(context["disclaimer"], True))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526476"))
        canvas.drawString(20 * mm, 10 * mm, f"NammaRetina | Report #{report_id} | Format v{context['format_version']}")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    # Publish a complete PDF atomically, even when requests overlap.
    fd, temporary = tempfile.mkstemp(prefix=f".report_{report_id}_", suffix=".pdf", dir=out_dir)
    os.close(fd)
    try:
        doc = SimpleDocTemplate(temporary, pagesize=A4, leftMargin=20 * mm,
                                rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=20 * mm,
                                title=f"NammaRetina Report {report_id}", author="NammaRetina",
                                subject=f"Standardized screening report v{context['format_version']}")
        doc.build(elements, onFirstPage=footer, onLaterPages=footer)
        os.replace(temporary, pdf_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    logger.info("Standardized PDF generated for report %s", report_id)
    return str(pdf_path)
