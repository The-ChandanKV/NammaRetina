# NammaRetina

**AI-Based Multi-Agent Diabetic Retinopathy Diagnostic and Progression Analysis System**

NammaRetina is an end-to-end system for automated diabetic retinopathy (DR) detection, severity grading, explainable AI visualisation, progression tracking, disease spread simulation, treatment recommendation, LLM-powered report interpretation, and PDF report generation — all accessible through a Flask web application with a doctor dashboard.

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd NammaRetina
pip install -r requirements.txt
```

### 2. Set up environment variables (optional)

Create a `.env` file in the project root with your LLM API key(s). The system tries providers in this order and falls back to rule-based explanations if none are set:

```env
# Google Gemini (recommended — already referenced in config.py)
GEMINI_API_KEY=your-gemini-api-key

# OR OpenAI
OPENAI_API_KEY=your-openai-api-key

# OR HuggingFace
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

### 3. Place the trained model (optional)

If you have the trained EfficientNetB0 model file, place it at:

```
models/efficientnetb0_dr.keras
```

The app works without it (using fallback predictions with 0% confidence), but real predictions require the model.

### 4. Run the application

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Full Pipeline

When you upload a retinal image through the web interface, the following pipeline runs automatically:

1. **Image Upload** — Retinal fundus photograph is saved to `static/uploads/`
2. **Patient Registration** — Patient details are stored in the SQLite database
3. **Model Prediction** — EfficientNetB0 classifies DR severity (0–4)
4. **Grad-CAM Heatmap** — Highlights the retinal regions influencing the prediction
5. **Progression Tracking** — Compares current severity against prior scans
6. **Disease Spread Simulation** — Probabilistic visualisation of future retinal changes
7. **Treatment Recommendation** — Rule-based clinical follow-up guidance
8. **LLM Explanation** — Plain-language summary for doctors and patients
9. **Database Record** — Full report saved to SQLite
10. **PDF Report** — Downloadable medical report generated via ReportLab

---

## Web Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Upload form with patient details |
| `/upload` | POST | Runs the full diagnostic pipeline |
| `/result/<report_id>` | GET | Displays the diagnostic result |
| `/dashboard` | GET | Doctor dashboard — all patients and reports |
| `/download/<report_id>` | GET | Downloads the PDF report |

---

## Project Structure

```
NammaRetina/
├── app.py                     # Flask web application (Phase 10)
├── config.py                  # Central configuration
├── database.py                # SQLite database layer
├── model_loader.py            # Model loading and prediction
├── gradcam.py                 # Grad-CAM heatmap generation (Phase 5)
├── preprocess_local.py        # Image preprocessing pipeline (Phase 2)
├── progression.py             # Progression tracking (Phase 6)
├── simulation.py              # Disease spread simulation (Phase 7)
├── treatment.py               # Treatment recommendations (Phase 8)
├── llm_agent.py               # LLM integration (Phase 9)
├── report_generator.py        # PDF report generation (Phase 12)
├── requirements.txt           # Python dependencies
├── database.db                # SQLite database
├── APTOS-19/                  # Dataset metadata (CSVs)
├── Colab Notebooks/           # Training and augmentation notebooks
├── models/                    # Trained model files (.keras)
├── static/
│   ├── uploads/               # Uploaded retinal images
│   ├── heatmaps/              # Grad-CAM heatmap outputs
│   ├── reports/               # Generated PDF reports
│   └── simulations/           # Simulation outputs
├── reports/                   # Progression graphs
└── templates/
    ├── index.html             # Upload page
    ├── result.html            # Result display page
    └── dashboard.html         # Doctor dashboard
```

---

## Implementation Phases

| Phase | Component | Module |
|---|---|---|
| 1 | Dataset Preparation | `APTOS-19/` |
| 2 | Preprocessing Pipeline | `preprocess_local.py` |
| 3 | CNN Model (EfficientNetB0) | `Colab Notebooks/` |
| 4 | Model Training | `Colab Notebooks/` |
| 5 | Grad-CAM Integration | `gradcam.py` |
| 6 | Progression Tracking | `progression.py` |
| 7 | Disease Spread Simulation | `simulation.py` |
| 8 | Treatment Recommendation | `treatment.py` |
| 9 | LLM Integration | `llm_agent.py` |
| 10 | Flask Web Application | `app.py` |
| 11 | Doctor Dashboard | `templates/dashboard.html` |
| 12 | Report Generation | `report_generator.py` |

---

## DR Severity Levels

| Stage | Label | Recommendation |
|---|---|---|
| 0 | No DR | Routine yearly screening |
| 1 | Mild | Monitor blood sugar and follow-up |
| 2 | Moderate | Ophthalmologist consultation advised |
| 3 | Severe | Immediate specialist consultation |
| 4 | Proliferative DR | Urgent retinal treatment required |

---

## Technology Stack

- **Python**, **TensorFlow / Keras**, **EfficientNetB0**
- **OpenCV**, **Pillow**, **Matplotlib**
- **Flask**, **SQLite**
- **ReportLab** (PDF generation)
- **Google Gemini / OpenAI / HuggingFace** (LLM integration)

---

## Team

g7 — Pranav, Chandan, Jyotir, Jhenka
