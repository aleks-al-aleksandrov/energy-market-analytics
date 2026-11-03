"""Central project configuration: paths, zones, period, weather points, seeds."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models"
for _p in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "Europe/Sofia"
SEED = 42

# --- Period -----------------------------------------------------------------
# History starts 2023-01-01 (local). End = start of today (local) -> last complete day.
START = pd.Timestamp("2023-01-01", tz=LOCAL_TZ)


def default_end() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LOCAL_TZ).normalize()


# --- Bidding zones ----------------------------------------------------------
FOCUS_ZONE = "BG"
NEIGHBOUR_ZONES = ["RO", "GR", "HU"]
PRICE_ZONES = [FOCUS_ZONE] + NEIGHBOUR_ZONES
# Physical cross-border flows BG <-> neighbours (ENTSO-E country codes)
FLOW_NEIGHBOURS = ["RO", "GR", "RS", "MK", "TR"]

# --- Weather points (Open-Meteo) --------------------------------------------
WEATHER_POINTS = {
    "sofia": (42.70, 23.32),        # demand centre
    "dobrich": (43.57, 27.83),      # NE coast, main wind cluster (Kaliakra)
    "stara_zagora": (42.42, 25.63), # sunny Thracian plain, solar cluster
}
WEATHER_VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]

# --- Validation -------------------------------------------------------------
HOLDOUT_MONTHS = 3            # final test set, never used for tuning
FIRST_TEST_MONTH = "2024-07"  # first walk-forward test month (>= 12 months of training before)

# --- Price events -----------------------------------------------------------
SPIKE_QUANTILE = 0.95

