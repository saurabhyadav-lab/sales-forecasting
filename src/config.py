"""Step 1-3: Problem definition and project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# --- Problem definition (Step 1) ---
TARGET = "sales"          # what we forecast
FREQ = "D"                # daily data
HORIZON = 30              # forecast the next 30 days
TEST_DAYS = 60            # hold-out window for final evaluation
SEASON = 7                # weekly seasonality for daily data
EXOG = ["promo", "holiday", "open"]   # known-in-advance external variables
SEED = 42

for _d in (DATA_RAW, DATA_PROC, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
