"""
NammaRetina - Configuration Module
Central configuration for all system parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Path Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "APTOS-19")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
HEATMAP_DIR = os.path.join(STATIC_DIR, "heatmaps")
REPORT_DIR = os.path.join(STATIC_DIR, "reports")
SIMULATION_DIR = os.path.join(STATIC_DIR, "simulations")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# =============================================================================
# Dataset Configuration
# =============================================================================
TRAIN_CSV = os.path.join(DATASET_DIR, "train_1.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_images", "train_images")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "val_images", "val_images")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test_images", "test_images")

# =============================================================================
# Model Configuration
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
MODEL_NAME = "efficientnetb0_dr.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# DR Severity Labels
DR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# =============================================================================
# Treatment Recommendations (Rule-Based)
# =============================================================================
TREATMENT_RECOMMENDATIONS = {
    0: {
        "severity": "No DR",
        "recommendation": "Routine yearly screening recommended.",
        "urgency": "low",
        "follow_up": "12 months"
    },
    1: {
        "severity": "Mild",
        "recommendation": "Monitor blood sugar levels closely. Follow-up in 6-9 months.",
        "urgency": "low",
        "follow_up": "6-9 months"
    },
    2: {
        "severity": "Moderate",
        "recommendation": "Ophthalmologist consultation advised. Close monitoring required.",
        "urgency": "medium",
        "follow_up": "3-6 months"
    },
    3: {
        "severity": "Severe",
        "recommendation": "Immediate specialist consultation required. Risk of vision loss.",
        "urgency": "high",
        "follow_up": "1-3 months"
    },
    4: {
        "severity": "Proliferative DR",
        "recommendation": "Urgent retinal treatment required. Immediate intervention needed.",
        "urgency": "critical",
        "follow_up": "Immediate"
    }
}

# =============================================================================
# Flask Configuration
# =============================================================================
# NOTE: the app uses raw sqlite3 against database.db (see database.py), not
# SQLAlchemy — so no SQLALCHEMY_* settings are defined here.
_DEFAULT_SECRET_KEY = "nammaretina-secret-key-change-in-production"
SECRET_KEY = os.getenv("SECRET_KEY", _DEFAULT_SECRET_KEY)
# True when no SECRET_KEY was provided via the environment. app.py warns on this
# because sessions (and therefore logins) are insecure with the shared default.
SECRET_KEY_IS_DEFAULT = SECRET_KEY == _DEFAULT_SECRET_KEY
# Debug is OFF unless FLASK_DEBUG is explicitly truthy — never ship debug=True.
DEBUG = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

# =============================================================================
# LLM Configuration
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-1.5-flash"

# =============================================================================
# Grad-CAM Configuration
# =============================================================================
GRADCAM_LAYER_NAME = "top_conv"  # EfficientNetB0 last conv layer
GRADCAM_COLORMAP = "jet"

# =============================================================================
# Ensure directories exist
# =============================================================================
for directory in [MODEL_DIR, UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR, SIMULATION_DIR]:
    os.makedirs(directory, exist_ok=True)
