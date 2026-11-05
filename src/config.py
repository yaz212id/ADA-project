"""Project-wide paths and sample dates.

Every notebook imports these values, so the folders and the sample period are
defined in one place only.
"""
from pathlib import Path

# Repository root = the folder that contains src/
ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_XLS_DIR = RAW_DIR / "xls"         # original EIA .xls downloads: NOT committed (CSV copies are)
RAW_PRICES_DIR = RAW_DIR / "prices"   # EIA spot prices as CSV: NOT committed (third-party data)
RAW_WEEKLY_DIR = RAW_DIR / "weekly"   # EIA Weekly Petroleum Status Report as CSV: committed
MANIFEST_PATH = RAW_DIR / "manifest.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# Outputs used in the paper
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Sample definition. The end date is fixed so that results do not change
# when EIA publishes new weeks. 2026-10-30 is a Friday (end of an EIA week).
SAMPLE_START = "2000-01-01"
SAMPLE_END = "2026-10-30"

# Chronological split (train / tune / test), as in the proposal
TRAIN_END = "2014-12-31"   # train: 2000-2014
VALID_END = "2018-12-31"   # tune:  2015-2018, test: 2019-SAMPLE_END
