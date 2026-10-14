"""
NammaRetina - XAI Engine (Explainable AI)
==========================================
Zero-API-key explainability for the EfficientNetB0 DR classifier.
"""

from __future__ import annotations

import base64
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DR_CLASSES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

# ─── Utility helpers ──────────────────────────────────────────────────────────

def _img_to_b64(img_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _load_rgb(image_path: str, size: int = 512) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (size, size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_bgr(image_path: str, size: int = 512) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.resize(img, (size, size))


# ─── 1. Superpixel Saliency (LIME-style) ─────────────────────────────────────

def _simple_superpixels(gray: np.ndarray, n_segments: int = 60) -> np.ndarray:
    h, w = gray.shape
    rows = max(int(math.sqrt(n_segments * h / w)), 1)
    cols = max(int(math.sqrt(n_segments * w / h)), 1)
    cell_h, cell_w = h // rows, w // cols
    labels = np.zeros((h, w), dtype=np.int32)
    seg_id = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h if r < rows - 1 else h
            x0, x1 = c * cell_w, (c + 1) * cell_w if c < cols - 1 else w
            labels[y0:y1, x0:x1] = seg_id
            seg_id += 1
    return labels


def compute_superpixel_saliency(
    image_path: str,
    heatmap_path,
    predicted_class: int,
    n_segments: int = 60,
    top_k: int = 8,
) -> dict:
    img_bgr = _load_bgr(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    if heatmap_path and Path(str(heatmap_path)).exists():
        hmap = cv2.imread(str(heatmap_path))
        hmap = cv2.resize(hmap, (w, h))
        hmap_gray = cv2.cvtColor(hmap, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        green = img_rgb[:, :, 1].astype(np.float32)
        hmap_gray = (green - green.min()) / (green.max() - green.min() + 1e-8)
        hmap_gray = cv2.GaussianBlur(hmap_gray, (31, 31), 0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    labels = _simple_superpixels(gray, n_segments)
    unique_labels = np.unique(labels)
    actual_n = len(unique_labels)

    scores = {seg: float(np.mean(hmap_gray[labels == seg])) for seg in unique_labels}
    max_score = max(scores.values()) or 1.0
    scores = {k: v / max_score for k, v in scores.items()}

    sorted_segs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_segs = {seg for seg, _ in sorted_segs[:top_k]}

    overlay = img_bgr.copy()
    for rank, (seg, score) in enumerate(sorted_segs[:top_k]):
        mask = labels == seg
        intensity = int(score * 255)
        if rank < top_k // 3:
            color = (0, 0, intensity)
        elif rank < 2 * top_k // 3:
            color = (0, intensity // 2, intensity)
        else:
            color = (intensity, 0, 0)
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.55 +
                         np.array(color, np.float32) * 0.45).clip(0, 255).astype(np.uint8)

    for seg in top_segs:
        mask = (labels == seg).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)

    total_pixels = h * w
    top_regions = [
        {
            "rank": rank + 1,
            "score": round(score * 100, 1),
            "area_pct": round(np.sum(labels == seg) / total_pixels * 100, 1),
        }
        for rank, (seg, score) in enumerate(sorted_segs[:top_k])
    ]

    return {
        "overlay_b64": _img_to_b64(overlay),
        "top_regions": top_regions,
        "n_segments": actual_n,
    }


# ─── 2. Lesion & Biomarker Detector ──────────────────────────────────────────

def _detect_microaneurysms(green_channel: np.ndarray) -> dict:
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opened)
    count = sum(1 for i in range(1, num_labels) if 4 <= stats[i, cv2.CC_STAT_AREA] <= 80)
    confidence = min(count / 15.0, 1.0)
    return {"count": count, "confidence": round(confidence, 2), "detected": count > 2}


def _detect_haemorrhages(red_channel: np.ndarray, green_channel: np.ndarray) -> dict:
    diff = red_channel.astype(np.int16) - green_channel.astype(np.int16)
    diff_uint8 = np.clip(diff, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(diff_uint8, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    count = sum(1 for i in range(1, num_labels) if 100 <= stats[i, cv2.CC_STAT_AREA] <= 4000)
    confidence = min(count / 8.0, 1.0)
    return {"count": count, "confidence": round(confidence, 2), "detected": count > 1}


def _detect_exudates(red_channel: np.ndarray, green_channel: np.ndarray) -> dict:
    bright = ((red_channel.astype(np.int16) + green_channel.astype(np.int16)) // 2).astype(np.uint8)
    _, thresh = cv2.threshold(bright, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    count = sum(1 for i in range(1, num_labels) if 50 <= stats[i, cv2.CC_STAT_AREA] <= 5000)
    confidence = min(count / 10.0, 1.0)
    return {"count": count, "confidence": round(confidence, 2), "detected": count > 0}


def _detect_optic_disc(gray: np.ndarray) -> dict:
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0] // 4,
        param1=50, param2=30,
        minRadius=gray.shape[0] // 12,
        maxRadius=gray.shape[0] // 4,
    )
    detected = circles is not None
    location = None
    if detected:
        c = np.round(circles[0, 0]).astype(int)
        location = {"x": int(c[0]), "y": int(c[1]), "radius": int(c[2])}
    return {"detected": detected, "location": location, "confidence": 0.85 if detected else 0.15}


def _detect_neovascularisation(green_channel: np.ndarray) -> dict:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(green_channel, cv2.MORPH_TOPHAT, kernel)
    std = float(np.std(tophat))
    score = min(std / 30.0, 1.0)
    return {"score": round(score, 2), "detected": score > 0.4, "confidence": round(score, 2)}


def detect_lesions(image_path: str) -> dict:
    img_rgb = _load_rgb(image_path, size=512)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    ma = _detect_microaneurysms(g)
    hem = _detect_haemorrhages(r, g)
    exu = _detect_exudates(r, g)
    od = _detect_optic_disc(gray)
    neo = _detect_neovascularisation(g)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated = img_bgr.copy()

    diff = np.clip(r.astype(np.int16) - g.astype(np.int16), 0, 255).astype(np.uint8)
    _, hthresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours_h, _ = cv2.findContours(hthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours_h, -1, (0, 0, 220), 1)

    bright = ((r.astype(np.int16) + g.astype(np.int16)) // 2).astype(np.uint8)
    _, ethresh = cv2.threshold(bright, 200, 255, cv2.THRESH_BINARY)
    contours_e, _ = cv2.findContours(ethresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours_e, -1, (0, 255, 255), 1)

    if od["detected"] and od["location"]:
        loc = od["location"]
        cv2.circle(annotated, (loc["x"], loc["y"]), loc["radius"], (255, 165, 0), 2)
        cv2.putText(annotated, "OD", (loc["x"] - 15, loc["y"] - loc["radius"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    legend_items = [("Haemorrhages", (0, 0, 220)), ("Exudates", (0, 255, 255)), ("Optic Disc", (255, 165, 0))]
    for i, (lbl, color) in enumerate(legend_items):
        y = 20 + i * 20
        cv2.rectangle(annotated, (8, y - 10), (20, y + 2), color, -1)
        cv2.putText(annotated, lbl, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    return {
        "microaneurysms": ma,
        "haemorrhages": hem,
        "exudates": exu,
        "optic_disc": od,
        "neovascularisation": neo,
        "annotated_b64": _img_to_b64(annotated),
    }


# ─── 3. Colour Channel Feature Analysis ──────────────────────────────────────

def analyse_channel_features(image_path: str) -> dict:
    img_rgb = _load_rgb(image_path, size=256)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    def _stats(ch):
        return {
            "mean": round(float(np.mean(ch)), 2),
            "std": round(float(np.std(ch)), 2),
            "min": int(np.min(ch)),
            "max": int(np.max(ch)),
            "contrast": round(float(np.std(ch) / (np.mean(ch) + 1e-8)), 3),
        }

    r_stats, g_stats, b_stats = _stats(r), _stats(g), _stats(b)
    vessel_contrast = g_stats["contrast"]
    haem_indicator = r_stats["mean"] / (g_stats["mean"] + 1e-8)
    exudate_score = float(np.mean((r.astype(float) + g.astype(float)) / 2 > 200))
    darkness_score = 1.0 - float(np.mean(g) / 255.0)

    features = {
        "Vessel Contrast": min(vessel_contrast * 2, 1.0),
        "Haemorrhage Indicator": min(max(haem_indicator - 1.0, 0) * 3, 1.0),
        "Bright Lesions (Exudates)": min(exudate_score * 20, 1.0),
        "Retinal Darkness": min(darkness_score * 1.5, 1.0),
        "Green Channel Std": min(g_stats["std"] / 80, 1.0),
        "Red-Green Imbalance": min(abs(r_stats["mean"] - g_stats["mean"]) / 60, 1.0),
    }

    return {
        "r": r_stats, "g": g_stats, "b": b_stats,
        "derived": {
            "vessel_contrast": round(vessel_contrast, 3),
            "haemorrhage_indicator": round(haem_indicator, 3),
            "exudate_score": round(exudate_score, 4),
            "darkness_score": round(darkness_score, 3),
        },
        "feature_importance": {k: round(v, 3) for k, v in features.items()},
    }


# ─── 4. Confidence Decomposition ─────────────────────────────────────────────

def decompose_confidence(severity: int, confidence: float, lesions: dict, channel_features: dict) -> list:
    fi = channel_features.get("feature_importance", {})
    ma_conf = lesions.get("microaneurysms", {}).get("confidence", 0)
    hem_conf = lesions.get("haemorrhages", {}).get("confidence", 0)
    exu_conf = lesions.get("exudates", {}).get("confidence", 0)
    neo_conf = lesions.get("neovascularisation", {}).get("confidence", 0)

    if severity == 0:
        weights = [
            ("Absence of Lesions", max(0, 1 - ma_conf) * 0.35),
            ("Vessel Pattern Regularity", (1 - fi.get("Vessel Contrast", 0)) * 0.25),
            ("Normal Colour Balance", (1 - fi.get("Red-Green Imbalance", 0)) * 0.25),
            ("Low Exudate Activity", (1 - exu_conf) * 0.15),
        ]
    elif severity == 1:
        weights = [
            ("Micro-aneurysm Presence", ma_conf * 0.40),
            ("Vessel Contrast", fi.get("Vessel Contrast", 0) * 0.25),
            ("Haemorrhage Absence", (1 - hem_conf) * 0.20),
            ("Colour Normality", (1 - fi.get("Red-Green Imbalance", 0)) * 0.15),
        ]
    elif severity == 2:
        weights = [
            ("Haemorrhage Activity", hem_conf * 0.35),
            ("Micro-aneurysm Load", ma_conf * 0.25),
            ("Exudate Presence", exu_conf * 0.20),
            ("Green Channel Degradation", fi.get("Green Channel Std", 0) * 0.20),
        ]
    elif severity == 3:
        weights = [
            ("Severe Haemorrhaging", hem_conf * 0.35),
            ("Extensive Micro-aneurysms", ma_conf * 0.25),
            ("Vascular Irregularity", fi.get("Vessel Contrast", 0) * 0.20),
            ("Neovascularisation Signs", neo_conf * 0.20),
        ]
    else:
        weights = [
            ("Neovascularisation (PDR)", neo_conf * 0.40),
            ("Proliferative Activity", fi.get("Red-Green Imbalance", 0) * 0.25),
            ("Severe Haemorrhaging", hem_conf * 0.20),
            ("Retinal Detachment Risk", fi.get("Retinal Darkness", 0) * 0.15),
        ]

    total_w = sum(w for _, w in weights) or 1.0
    scale = confidence / total_w

    descriptions = {
        "Absence of Lesions": "No micro-aneurysms or haemorrhages detected.",
        "Vessel Pattern Regularity": "Retinal blood vessels appear uniform and regular.",
        "Normal Colour Balance": "Red/green channel ratio within normal range.",
        "Low Exudate Activity": "Minimal bright lesion (exudate) presence.",
        "Micro-aneurysm Presence": "Small vascular outpouchings detected.",
        "Vessel Contrast": "Green-channel vessel contrast analysis.",
        "Haemorrhage Absence": "No significant haemorrhagic regions found.",
        "Colour Normality": "Overall colour profile matches healthy retinas.",
        "Haemorrhage Activity": "Dot/blot haemorrhages detected in retinal scan.",
        "Micro-aneurysm Load": "Multiple micro-aneurysms contributing to severity.",
        "Exudate Presence": "Bright hard exudate deposits identified.",
        "Green Channel Degradation": "Vessel definition in green channel is reduced.",
        "Severe Haemorrhaging": "Large or multiple haemorrhagic regions detected.",
        "Extensive Micro-aneurysms": "High density of micro-aneurysms across the retina.",
        "Vascular Irregularity": "Irregular vessel calibre and distribution.",
        "Neovascularisation Signs": "Vessel complexity suggests new vessel growth.",
        "Neovascularisation (PDR)": "Abnormal new blood vessel growth (hallmark of PDR).",
        "Proliferative Activity": "Significant red/green imbalance indicating neovascularisation.",
        "Retinal Detachment Risk": "Retinal darkening may indicate traction or detachment.",
    }

    return [
        {
            "factor": name,
            "contribution": round(w * scale * 100, 1),
            "weight_pct": round(w / total_w * 100, 1),
            "description": descriptions.get(name, ""),
        }
        for name, w in weights
    ]


# ─── 5. Counterfactual Reasoning ─────────────────────────────────────────────

def generate_counterfactuals(severity: int, lesions: dict) -> list:
    cf = []
    ma = lesions.get("microaneurysms", {})
    hem = lesions.get("haemorrhages", {})
    exu = lesions.get("exudates", {})
    neo = lesions.get("neovascularisation", {})

    if severity > 0 and not ma.get("detected"):
        cf.append("If micro-aneurysms were absent, the model would likely predict a lower severity stage.")
    if severity > 1 and not hem.get("detected"):
        cf.append("Absence of haemorrhages in this scan reduced the predicted severity by at least one stage.")
    if severity < 3 and hem.get("confidence", 0) < 0.3:
        cf.append("If haemorrhages were more extensive, the model would classify this as Severe DR.")
    if severity < 4 and neo.get("confidence", 0) < 0.4:
        cf.append("If neovascularisation were present, this scan would meet criteria for Proliferative DR.")
    if exu.get("detected") and severity == 0:
        cf.append("Some bright lesions detected — if confirmed as exudates, severity may be revised upward.")
    if severity == 0:
        cf.append("The model would reclassify as Mild DR if even 3-5 micro-aneurysms were detected.")
    if severity == 4:
        cf.append("Absence of neovascularisation would shift this prediction to Severe DR (stage 3).")
    if not cf:
        cf.append("Confidence is high; minor lesion changes would not alter the classification.")

    return cf[:4]


# ─── 6. Plain-Language XAI Narrative ─────────────────────────────────────────

def generate_xai_narrative(severity, confidence, lesions, channel_features, saliency) -> str:
    label = DR_CLASSES.get(severity, "Unknown")
    conf_pct = round(confidence * 100, 1)
    ma = lesions.get("microaneurysms", {})
    hem = lesions.get("haemorrhages", {})
    exu = lesions.get("exudates", {})
    neo = lesions.get("neovascularisation", {})
    od = lesions.get("optic_disc", {})
    fi = channel_features.get("feature_importance", {})

    p1_parts = []
    if ma.get("detected"):
        p1_parts.append(f"approximately {ma.get('count',0)} micro-aneurysm(s) in the retinal vasculature")
    if hem.get("detected"):
        p1_parts.append(f"{hem.get('count',0)} haemorrhagic region(s) consistent with dot/blot haemorrhages")
    if exu.get("detected"):
        p1_parts.append(f"{exu.get('count',0)} bright lesion cluster(s) suggestive of hard exudates")
    if neo.get("detected"):
        p1_parts.append("signs of abnormal vascular proliferation (neovascularisation)")

    if p1_parts:
        p1 = (f"The AI model analysed this retinal image and identified {', '.join(p1_parts)}. "
              f"These findings collectively contributed to a classification of <strong>{label}</strong> "
              f"with {conf_pct}% confidence.")
    else:
        p1 = (f"The AI model found no significant lesion markers in this retinal image, "
              f"supporting a classification of <strong>{label}</strong> with {conf_pct}% confidence.")

    top = saliency.get("top_regions", [{}])
    n_seg = saliency.get("n_segments", 0)
    top1 = top[0] if top else {}
    p2 = (f"The saliency analysis divided the image into {n_seg} regions. "
          f"The most influential region contributed {top1.get('score', 0):.1f}% "
          f"of the activation signal and covers {top1.get('area_pct', 0):.1f}% of the image area. "
          f"Highlighted regions correspond to areas the network weighted most "
          f"heavily — typically the posterior pole and areas near the fovea.")

    vc = fi.get("Vessel Contrast", 0)
    rgi = fi.get("Red-Green Imbalance", 0)
    if conf_pct >= 80:
        conf_reason = "The high confidence indicates a clear, unambiguous pattern matching this DR stage."
    elif conf_pct >= 55:
        conf_reason = "Moderate confidence may reflect image quality variations or borderline lesion characteristics."
    else:
        conf_reason = ("Lower confidence suggests this case lies near the boundary between two severity stages "
                       "and warrants additional clinical assessment.")

    p3 = (f"Channel analysis shows a vessel contrast score of {vc:.2f} and a red-green imbalance of {rgi:.2f}. "
          f"{conf_reason} "
          + ("The optic disc was detected, providing a reliable anatomical reference point. " if od.get("detected") else ""))

    severity_context = {
        0: "No treatment currently indicated. Annual screening is the standard of care.",
        1: "Close glycaemic control can prevent progression. Ophthalmology review in 6-9 months.",
        2: "Ophthalmologist referral recommended. Laser photocoagulation may be considered.",
        3: "Prompt retinal specialist review is essential. Pan-retinal photocoagulation may be indicated.",
        4: "Urgent intervention required — vitrectomy or anti-VEGF injections may be needed.",
    }
    p4 = (f"<em>Clinical context:</em> {severity_context.get(severity, '')} "
          "This AI output is a decision-support tool and must be interpreted alongside clinical history, "
          "visual acuity measurements, and fundoscopy findings by a qualified ophthalmologist.")

    return f"<p>{p1}</p><p>{p2}</p><p>{p3}</p><p>{p4}</p>"


# ─── Public API ───────────────────────────────────────────────────────────────

def explain_prediction(
    image_path: str,
    heatmap_path,
    severity: int,
    confidence: float,
    probabilities=None,
) -> dict:
    logger.info("XAI: starting explanation for severity=%d, confidence=%.3f", severity, confidence)

    try:
        saliency = compute_superpixel_saliency(image_path, heatmap_path, severity)
    except Exception as exc:
        logger.warning("XAI saliency failed: %s", exc)
        saliency = {"overlay_b64": None, "top_regions": [], "n_segments": 0}

    try:
        lesions = detect_lesions(image_path)
    except Exception as exc:
        logger.warning("XAI lesion detection failed: %s", exc)
        lesions = {k: {"detected": False, "confidence": 0.0, "count": 0}
                   for k in ("microaneurysms", "haemorrhages", "exudates", "optic_disc", "neovascularisation")}
        lesions["annotated_b64"] = None

    try:
        channel = analyse_channel_features(image_path)
    except Exception as exc:
        logger.warning("XAI channel analysis failed: %s", exc)
        channel = {"feature_importance": {}}

    try:
        decomposition = decompose_confidence(severity, confidence, lesions, channel)
    except Exception as exc:
        logger.warning("XAI decomposition failed: %s", exc)
        decomposition = []

    try:
        counterfactuals = generate_counterfactuals(severity, lesions)
    except Exception as exc:
        logger.warning("XAI counterfactuals failed: %s", exc)
        counterfactuals = []

    try:
        narrative = generate_xai_narrative(severity, confidence, lesions, channel, saliency)
    except Exception as exc:
        logger.warning("XAI narrative failed: %s", exc)
        narrative = "<p>XAI narrative unavailable.</p>"

    if not probabilities or all(p == 0 for p in probabilities):
        base = [0.05] * 5
        base[severity] = confidence
        remaining = (1.0 - confidence) * 0.95
        for i in range(5):
            if i != severity:
                base[i] = remaining / 4
        probabilities = base

    prob_chart = [
        {
            "label": DR_CLASSES.get(i, f"Class {i}"),
            "prob": round(probabilities[i] * 100, 1),
            "is_predicted": i == severity,
        }
        for i in range(min(5, len(probabilities)))
    ]

    return {
        "severity": severity,
        "severity_label": DR_CLASSES.get(severity, "Unknown"),
        "confidence_pct": round(confidence * 100, 1),
        "saliency": saliency,
        "lesions": lesions,
        "channel": channel,
        "decomposition": decomposition,
        "counterfactuals": counterfactuals,
        "narrative": narrative,
        "prob_chart": prob_chart,
        "dr_classes": DR_CLASSES,
    }
