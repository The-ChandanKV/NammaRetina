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

Create a `.env` file in the project root using `.env.example`. Supabase credentials are required; LLM API keys are optional. The system tries providers in this order and falls back to rule-based explanations if none are set:

```env
# Google Gemini (recommended — already referenced in config.py)
GEMINI_API_KEY=your-gemini-api-key

# OR OpenAI
OPENAI_API_KEY=your-openai-api-key

# OR HuggingFace
HUGGINGFACE_API_KEY=your-huggingface-api-key

# Session signing key for login (set a long random value in production).
# If unset, a built-in development default is used and the app logs a warning.
SECRET_KEY=change-me-to-a-long-random-string
```

### 3. Place the trained model (optional)

Use [the corrected GPU training notebook](training/Train_NammaRetina_GPU.ipynb) and
[the training/deployment guide](training/README.md). The trainer downloads a versioned
APTOS derivative, quarantines exact duplicates, selects on validation accuracy, and
exports a compatible `efficientnetb0_dr.keras` with metrics and a SHA256 digest.

Place a selected model at `models/efficientnetb0_dr.keras`, or configure `MODEL_URL`
and `MODEL_SHA256` for a verified download on Render. The app can start without
weights, but cannot record diagnoses until a real model loads.

### 4. Build the public landing page (required once / after UI changes)

The public landing page lives in `landing-source/` and compiles to `static/landing/`:

```bash
cd landing-source
npm install
npm run build
cd ..
```

Docker builds this automatically during image creation. Locally, rebuild whenever you change landing components.

### 5. Run the application

```bash
python app.py
```

Then open **http://localhost:5001** — the public landing page loads at `/`. Use **Doctor Sign In** for the clinical app.

### 6. Create your account

The dashboard and all patient data are behind a login. On first run, open `/login` and use
**Account setup** to register. **The first account created becomes the admin.** After setup, an administrator must create additional
accounts using **Add account**. Google sign-in is available to provisioned users. Approved clinicians
share the care-team dashboard; patient assignments and organization isolation are not implemented.

> To run with the auto-reloader/debugger for local development, set `FLASK_DEBUG=1`. Debug mode is
> **off** by default so it is never accidentally enabled in production.

---

## Full Pipeline

When you upload a retinal image through the web interface, the following pipeline runs automatically:

1. **Image Upload** — Retinal fundus photograph is saved to `static/uploads/`
2. **Patient Registration** — Patient details are stored in Supabase
3. **Model Prediction** — EfficientNetB0 classifies DR severity (0–4)
4. **Grad-CAM Heatmap** — Highlights the retinal regions influencing the prediction
5. **Progression Tracking** — Compares current severity against prior scans
6. **Longitudinal Progression Analysis** — Descriptive severity timeline from recorded visits (not a clinical forecast); optional illustrative spatial attention overlay
7. **Treatment Recommendation** — Rule-based clinical follow-up guidance
8. **LLM Explanation** — Plain-language summary for doctors and patients
9. **Database Record** — Full report saved to Supabase
10. **PDF Report** — Downloadable medical report generated via ReportLab

---

## Web Routes

| Route | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | Public | Premium landing page (Vite build in `static/landing/`) |
| `/landing/<path>` | GET | Public | Landing JS/CSS/images only (no patient files) |
| `/login` | GET/POST | Public | Doctor sign-in (existing Flask auth) |
| `/register` | GET/POST | Public* | First-user bootstrap / admin account creation |
| `/logout` | POST | — | Sign out → returns to landing |
| `/scan` | GET | Required | Retinal upload form (endpoint name: `index`) |
| `/upload` | POST | Required | Runs the full diagnostic pipeline |
| `/result/<report_id>` | GET | Required | Displays the diagnostic result |
| `/dashboard` | GET | Required | Doctor dashboard — all patients and reports |
| `/download/<report_id>` | GET | Required | Downloads the PDF report |
| `/xai/<report_id>` | GET | Required | Grad-CAM + on-demand LIME explainability |
| `/xai/<report_id>/lime` | POST | Required | Generate genuine LIME for the predicted class |
| `/evaluation` | GET | Admin | Research model-evaluation dashboard |
| `/files/<subdir>/<filename>` | GET | Required | Serves uploaded images, heatmaps, and simulations |

\*Public registration is locked after the first admin account exists.

### Landing ↔ Flask integration

- Source: `landing-source/` (React + Vite + Tailwind)
- Build output: `static/landing/` (committed or produced in Docker)
- Doctor Sign In uses a real `<a href="/signin">` so Flask handles authentication
- `/static/uploads|heatmaps|reports|simulations` remain blocked; only `/static/css/*` and `/landing/*` are public

---

## Project Structure

```
NammaRetina/
├── app.py                     # Flask web application (Phase 10)
├── config.py                  # Central configuration
├── database.py                # Supabase database layer
├── model_loader.py            # Model loading and prediction
├── gradcam.py                 # Grad-CAM heatmap generation (Phase 5)
├── preprocess_local.py        # Image preprocessing pipeline (Phase 2)
├── progression.py             # Progression tracking (Phase 6)
├── progression_analysis.py    # Longitudinal severity analysis
├── simulation.py              # Illustrative spatial attention overlay
├── lime_explainer.py          # Genuine LIME (perturbation + Ridge surrogate)
├── model_evaluation.py        # Clinical research evaluation metrics
├── explainability_evaluation.py
├── evaluation_reports.py      # Load evaluation runs for admin UI
├── treatment.py               # Treatment recommendations (Phase 8)
├── llm_agent.py               # LLM integration (Phase 9)
├── reporting.py               # Shared examination snapshot and report format
├── report_generator.py        # Standardized PDF report generation
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
    ├── dashboard.html         # Doctor dashboard
    ├── login.html             # Sign-in page
    └── register.html          # Account creation page
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
- **Flask**, **Supabase Auth / Postgres / Storage**
- **ReportLab** (PDF generation)
- **Google Gemini / OpenAI / HuggingFace** (LLM integration)

---

## Team

g7 — Pranav, Chandan, Jyotir, Jhenka


## Regression checks and deployment

Run `.venv/bin/python -m unittest discover -s tests -v` (or `python` in your activated environment).

### Standardized patient reports

Every examination uses the same eight sections in the browser and PDF: patient information,
screening result, standardized interpretation, recommendation, history through that examination,
scan and Grad-CAM, illustrative spatial simulation, and supplementary AI explanation.
Missing images or supplementary text leave an explicit unavailable notice; they do not remove
sections. A valid recorded result can still be exported when its images are unavailable.

`reporting.py` builds the shared snapshot. Comparisons exclude later examinations, including
later reports on the same day, and ignore records without a valid prediction. Patient overview
and directory status use the same report-derived history instead of a potentially stale
Progression row. Stored AI prose is supplementary; the main interpretation uses recorded fields.

PDF format v3 uses `report_<id>_v3.pdf` locally and `reports/pdfs/report_<id>_v3.pdf` in Storage.
Opening Export PDF automatically builds this format if it is missing; old PDFs do not need a
database migration. Regenerate PDF explicitly rebuilds it. Text is escaped and wrapped, images
preserve their aspect ratio, and complete files replace the cache atomically. Optional AI-text,
PDF, or PDF-storage failures retain the saved examination and allow later export retries.
The suite mocks external services and must not write to production Supabase. Legacy SQLite `db_path`
arguments now fail explicitly rather than silently writing to the configured Supabase project.

Follow-up scans must be started from the patient record to preserve its ID. Entering patient details
on the new-scan form creates a new patient, even if another record has the same demographics.
Training data preparation and online inference share `retinal_preprocessing.py`.

All POST forms require a session CSRF token. Generated files under `/static/` are blocked; private
artifacts are accessed through authenticated routes or short-lived Supabase signed URLs. PDFs and
source images can be restored from storage after a deployment. Failed storage writes are surfaced.

XAI displays only recorded classifier scores. Older reports without probability metadata show it as
unavailable. Image-processing lesion indicators are heuristic measurements, not model attributions.
Grad-CAM failure is shown as unavailable and does not generate a synthetic attention map.

Render deploys the GitHub `main` branch. `/health` exposes the running Render commit and whether the
model is available; responses include `X-App-Revision`. The endpoint returns HTTP 503 when the database or model is unavailable, and HTTP 200 only when
both are available. It does not validate model quality. Set `MODEL_URL` or provide model weights to enable analysis.


## Progression analysis, LIME, and evaluation

### Longitudinal analysis vs forecasting

`progression_analysis.py` compares recorded ordinal DR grades (0–4) across dated examinations, including intervals and direction of change. Apparent improvement does **not** prove clinical recovery. No supervised progression-risk model is trained because the repository lacks patient-linked longitudinal outcomes with defined horizons.

The spatial overlay in `simulation.py` is labeled **Illustrative Spatial Simulation — Not a Clinical Forecast**. It remaps Grad-CAM attention with FOV masking; it does not invent lesions or claim future spread.

### Grad-CAM vs genuine LIME

- **Grad-CAM** (`gradcam.py`): class-specific CNN activation maps.
- **LIME** (`lime_explainer.py`): SLIC (or grid) superpixels → batched perturbations → Ridge surrogate on the black-box class score. Generated on demand via `/xai/<id>/lime` and cached under `static/reports/lime_<id>.*`.
- The older heatmap-ranked region panel is **not** LIME.

Highlighted regions are not confirmed pathological lesions. Agreement between Grad-CAM and LIME does not establish diagnostic correctness.

### Running independent evaluation

```bash
# From saved predictions (y_true + probabilities):
python model_evaluation.py --predictions path/to/preds.npz --dataset aptos --split test

# Import a prior training/results metrics JSON (partial fields only):
python model_evaluation.py --import-legacy training/results/cpu_metrics.json --dataset aptos_internal_validation
```

Runs are written to timestamped folders under `evaluation_runs/` (JSON, CSV, markdown). The admin page `/evaluation` displays completed runs only — never fabricated metrics. External validation remains pending until a non-overlapping scored dataset run is produced.

Bootstrap 95% CIs are image-level by default, or patient-level when `patient_ids` are supplied.

### Computational notes

LIME uses batched inference (`n_samples` default 160 in the UI, configurable). Prefer on-demand generation on Render CPU instances. Classification and Grad-CAM continue if LIME fails.

---

## Clinical Limitations and Research Status

Technical implementation, Grad-CAM/LIME visualizations, illustrative spatial overlays, and internal evaluation metrics do **not** establish clinical effectiveness, diagnostic reliability, or regulatory approval. NammaRetina is a screening-assistance research system; clinician review remains required. Do not treat highlighted regions as confirmed lesions or treat illustrative simulations as prognoses.

---

## Latest training experiment

A corrected, CPU-trained candidate achieved **82.77% accuracy on 354 held-out images**,
macro-F1 **0.618**, and quadratic-weighted kappa **0.840**. Predictions are unchanged
by serialization, and prediction/Grad-CAM checks passed. Severe and proliferative
class recalls remain low; this is a research baseline for GPU fine-tuning and external
validation. The old notebook score is not directly comparable because duplicate
quarantine changes evaluation membership.

See [measured results](training/results/cpu_metrics.json),
[the GPU notebook](training/Train_NammaRetina_GPU.ipynb), and
[the model promotion steps](training/README.md#render-production-steps).


## Project audit fixes (5 November 2026)

- Database list/history queries page through results instead of silently accepting Supabase's row limit. Storage existence checks search and page for the requested filename.
- Restored images remain usable when Storage URL signing is unavailable. Local file serving stays within the requested artifact category.
- Sign-in destinations survive password errors and Google sign-in. OAuth callbacks require a browser-bound PKCE verifier; obsolete bearer tokens in URLs are rejected. Google account provisioning policy is unchanged.
- Grad-CAM scalar maps are projected back through the exact inference crop into original-image coordinates. The RGB overlay and a separate grayscale `*_gradcam_attention.png` are persisted. Shared preprocessing produces the same classifier pixels as before.
- Illustrative simulations use genuine scalar attention, reject RGB overlays and empty maps, and use versioned `*_simulation_v2.png` caches. Legacy attention is recomputed when possible; no synthetic central hotspot is substituted. These overlays remain illustrative, not forecasts.
- Model initialization is serialized across worker threads. Temporary load/download failures can recover after a 30-second backoff. Images exceeding 20 megapixels are rejected before inference to bound decoding memory.
- Supplementary fallback text describes predicted grades rather than claiming detected lesions or biological recovery. Remote text providers have explicit timeouts.
- Patient-cluster bootstrap repeats all observations for each sampled patient, including repeated draws. Undefined statistics remain null. Evaluation input validation rejects empty/noninteger labels and invalid prediction archives; NPZ files cannot unpickle Python objects.
- Evaluation run paths are sanitized and unique. Metrics publish atomically after the other artifacts, and the dashboard skips corrupt newer runs. Legacy imports retain their actual split and leave missing metrics unavailable.
- Constant-map diagnostics handle undefined correlation, deletion AUC works across supported NumPy versions, and invalid LIME parameters/cache metadata are rejected.
- The patient-entry CLI uses the current Supabase helpers. Legacy SQLite/simulation test entry points now run the current isolated tests rather than writing demo data or asserting obsolete simulation behavior.
- Frontend tests exercise the actual Flask SPA, with working lint and type checks. Patched development dependencies are locked, including a minimatch override for the known glob-expansion advisory. The build includes type checking.

Audit checks use synthetic records and mocked services; the saved classifier is also smoke-tested with a synthetic image. They do not constitute a manual review of every live patient record or a new clinical accuracy evaluation. Existing experiment result files and model weights are preserved.
