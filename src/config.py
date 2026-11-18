"""Central configuration for the uncertainty-in-asset-allocation project.

Keeping paths, asset definitions and simulation defaults in one place avoids
the previous duplication between ``prepare_data.py`` and ``app.py``.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HDF5_FILE = DATA_DIR / "asset_allocation_uncertainty.h5"
RETURNS_CSV_FILE = DATA_DIR / "monthly_returns.csv"

TITLE = "Az eszközallokáció input adataiban rejlő bizonytalanság"

# Short page title for the dashboard. The longer TITLE is kept as the HDF5
# metadata description so the data bundle remains self-describing.
APP_TITLE = "Eszközallokáció bizonytalansága"

# Asset universe. The dict order defines the column order everywhere.
TICKERS = {
    "USD Készpénz (BIL)": "BIL",
    "USA Állampapír (IEF)": "IEF",
    "USA Váll. Kötvény (LQD)": "LQD",
    "Arany (GLD)": "GLD",
}
ASSET_CLASSES = list(TICKERS.keys())
TICKER_LIST = list(TICKERS.values())

# Special portfolios computed in every simulation.
SPECIAL_KEYS = ("min_risk", "tangency", "risk_parity")

# Simulation defaults.
RISK_FREE_RATE = 0.02
CONFIDENCE_LEVEL = 0.95
N_SIMULATIONS = 100
N_FRONTIER_POINTS = 20
SEED = 42
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# HDF5 schema version. Bump this whenever the on-disk layout changes.
SCHEMA_VERSION = 2

# Streamlit cache version. Bump this when the portfolio definitions change so
# a hot-reloaded app cannot serve point estimates computed with old math.
CACHE_VERSION = 2
