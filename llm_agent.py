"""
NammaRetina - LLM Agent (Phase 9)
Converts technical DR predictions into plain-language explanations
suitable for doctors and patients.

Provider priority:
    1. Google Gemini  (GEMINI_API_KEY)
    2. OpenAI         (OPENAI_API_KEY)
    3. HuggingFace    (HUGGINGFACE_API_KEY)
    4. Rule-based fallback (no key required)
"""

from __future__ import annotations

import logging
import os

from config import DR_CLASSES

logger = logging.getLogger(__name__)

# ─── Prompt Template ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a medical AI assistant for the NammaRetina diabetic retinopathy "
    "diagnostic system. Given a patient's retinal scan analysis results, "
    "produce a short, plain-language explanation (3-5 sentences) suitable "
    "for both a doctor and a patient. Include:\n"
    "1. What the scan shows (severity in everyday terms).\n"
    "2. The confidence level of the analysis.\n"
    "3. How the condition has changed compared to previous scans.\n"
    "4. A clear recommendation (e.g. see a specialist, routine follow-up).\n"
    "Use only the supplied grade, score and recorded comparison. Do not claim "
    "specific lesions, image quality, biological recovery, treatment success, or "
    "future risk. The model is a classifier, not a lesion detector. Confidence is "
    "not diagnostic certainty. A grade change does not prove biological progression. "
    "State that clinical review is required. Do NOT use bullet points or HTML."
)


def _build_user_prompt(severity: int, confidence: float, progression_status: str) -> str:
    severity_label = DR_CLASSES.get(severity, "Unknown")
    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)
    return (
        f"Severity: {severity_label} (stage {severity}/4)\n"
        f"Confidence: {confidence_pct}%\n"
        f"Progression status: {progression_status}"
    )


# ─── Provider Implementations ────────────────────────────────────────────────

def _call_gemini(user_prompt: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-1.5-flash"))
    response = model.generate_content(
        f"{_SYSTEM_PROMPT}\n\n{user_prompt}",
        request_options={"timeout": 20},
        generation_config=genai.GenerationConfig(
            max_output_tokens=300,
            temperature=0.4,
        ),
    )
    return response.text.strip()


def _call_openai(user_prompt: str) -> str:
    """Call OpenAI Chat Completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=20, max_retries=0)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _call_huggingface(user_prompt: str) -> str:
    """Call HuggingFace Inference API."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.getenv("HUGGINGFACE_API_KEY", ""), timeout=20)
    response = client.text_generation(
        prompt=f"{_SYSTEM_PROMPT}\n\n{user_prompt}\n\nExplanation:",
        model=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
        max_new_tokens=300,
        temperature=0.4,
    )
    return response.strip()


def _fallback_explanation(severity: int, confidence: float, progression_status: str) -> str:
    """Rule-based fallback when no LLM API key is available."""
    from reporting import normalize_report
    from treatment import get_treatment_recommendation

    report = normalize_report({"severity": severity, "confidence": confidence})
    if not report["model_used"]:
        return "No valid model prediction is available for this examination. A new analysis is required."
    descriptions = (
        f"The model classified this examination as {report['severity_label']} "
        f"(grade {report['severity']}/4), with a recorded confidence of {report['confidence_pct']:.1f}%. "
        "This is a screening prediction, not a confirmed diagnosis; confidence is not diagnostic certainty."
    )
    progression_text = {
        "Initial": "No prior valid examination is available for comparison.",
        "Stable": "The recorded grade is unchanged from the previous examination.",
        "Worsened": "The recorded grade increased from the previous examination.",
        "Improved": "The recorded grade decreased from the previous examination.",
    }.get(progression_status, "No valid comparison is available.")
    return (f"{descriptions} {progression_text} "
            "A change in predicted grade does not prove biological progression or recovery. "
            f"{get_treatment_recommendation(report['severity'])}. A qualified clinician must review the result.")


# ─── Public API ───────────────────────────────────────────────────────────────

def explain_report(
    severity: int,
    confidence: float,
    progression_status: str,
) -> str:
    """Generate a plain-language explanation of a DR diagnostic report.

    Tries LLM providers in priority order and falls back to a rule-based
    explanation if no API key is available or the call fails.

    Parameters
    ----------
    severity : int
        DR severity class (0-4).
    confidence : float
        Model confidence score (0.0-1.0).
    progression_status : str
        One of "Initial", "Stable", "Worsened", "Improved".

    Returns
    -------
    str
        A short, plain-language explanation.
    """
    from reporting import normalize_report
    report = normalize_report({"severity": severity, "confidence": confidence})
    if not report["model_used"]:
        return _fallback_explanation(severity, confidence, progression_status)
    severity, confidence = report["severity"], report["confidence"]
    user_prompt = _build_user_prompt(severity, confidence, progression_status)

    # Try providers in priority order
    providers = []
    if os.getenv("GEMINI_API_KEY"):
        providers.append(("Gemini", _call_gemini))
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("OpenAI", _call_openai))
    if os.getenv("HUGGINGFACE_API_KEY"):
        providers.append(("HuggingFace", _call_huggingface))

    for name, call_fn in providers:
        try:
            logger.info("Calling %s for report explanation...", name)
            result = call_fn(user_prompt)
            if result:
                logger.info("Successfully received explanation from %s.", name)
                return result
        except Exception as exc:
            logger.warning("%s call failed: %s. Trying next provider.", name, exc)

    # Fallback
    logger.info("No LLM provider available — using rule-based explanation.")
    return _fallback_explanation(severity, confidence, progression_status)
