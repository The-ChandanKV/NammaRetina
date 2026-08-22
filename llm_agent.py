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
    "Do NOT use bullet points. Write in a warm, professional tone."
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
        generation_config=genai.GenerationConfig(
            max_output_tokens=300,
            temperature=0.4,
        ),
    )
    return response.text.strip()


def _call_openai(user_prompt: str) -> str:
    """Call OpenAI Chat Completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
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

    client = InferenceClient(token=os.getenv("HUGGINGFACE_API_KEY", ""))
    response = client.text_generation(
        prompt=f"{_SYSTEM_PROMPT}\n\n{user_prompt}\n\nExplanation:",
        model=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
        max_new_tokens=300,
        temperature=0.4,
    )
    return response.strip()


def _fallback_explanation(severity: int, confidence: float, progression_status: str) -> str:
    """Rule-based fallback when no LLM API key is available."""
    severity_label = DR_CLASSES.get(severity, "Unknown")
    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)

    # Severity-specific descriptions
    descriptions = {
        0: (
            "The retinal scan appears normal with no signs of diabetic retinopathy. "
            "The blood vessels and retinal structures look healthy."
        ),
        1: (
            "The retinal scan shows mild diabetic retinopathy with minor changes "
            "in the blood vessels, such as small micro-aneurysms."
        ),
        2: (
            "The retinal scan shows moderate diabetic retinopathy with visible "
            "abnormalities in the blood vessels, including dot and blot haemorrhages."
        ),
        3: (
            "The retinal scan shows severe diabetic retinopathy with significant "
            "vascular damage and a high risk of progression to vision-threatening stages."
        ),
        4: (
            "The retinal scan shows proliferative diabetic retinopathy, the most "
            "advanced stage, with abnormal new blood vessel growth that poses an "
            "immediate risk to vision."
        ),
    }

    description = descriptions.get(severity, "The retinal scan has been analysed.")

    # Progression context
    progression_phrases = {
        "Initial": "This is the first scan on record, so no comparison with previous visits is available.",
        "Stable": "Compared with previous scans, the condition appears stable with no significant change.",
        "Worsened": "Compared with previous scans, the condition has worsened, indicating disease progression.",
        "Improved": "Compared with previous scans, the condition has improved, which is an encouraging sign.",
    }
    progression_text = progression_phrases.get(
        progression_status,
        f"The current progression status is: {progression_status}.",
    )

    # Recommendations
    recommendations = {
        0: "A routine follow-up screening in 12 months is recommended.",
        1: "Monitoring blood sugar levels closely and a follow-up visit in 6-9 months is recommended.",
        2: "A consultation with an ophthalmologist is recommended for closer monitoring.",
        3: "Immediate consultation with a retinal specialist is strongly recommended to prevent vision loss.",
        4: "Urgent retinal treatment is required. Please seek immediate specialist intervention.",
    }
    recommendation = recommendations.get(severity, "Please consult your ophthalmologist.")

    return (
        f"{description} "
        f"The analysis confidence is {confidence_pct}%. "
        f"{progression_text} "
        f"{recommendation}"
    )


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
