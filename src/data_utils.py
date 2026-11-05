"""Download the EIA data, convert it to CSV and read it back.

EIA publishes the full history of each series as an Excel 97 (.xls) file at
https://www.eia.gov/dnav/pet/hist_xls/<SERIES_ID><frequency>.xls
(frequency: "d" = daily, "w" = weekly). No account or API key is needed.

The project keeps the data as one CSV per metric, named after the metric:
    data/raw/weekly/crude_stocks.csv, data/raw/prices/wti.csv, ...
Each CSV has two columns: date, <metric>.

Run from the repository root to convert .xls files you downloaded by hand:
    python -m src.data_utils "path/to/folder/with/xls/files"

Written with the help of Claude (Anthropic), which was asked to produce the
EIA download, conversion and reading code for this project (declared in the
paper appendix).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import MANIFEST_PATH, RAW_PRICES_DIR, RAW_WEEKLY_DIR, RAW_XLS_DIR, ROOT

EIA_HIST_URL = "https://www.eia.gov/dnav/pet/hist_xls/{series_id}{freq}.xls"
EIA_PAGE_URL = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s={series_id}&f={freq}"

# First bytes of every Excel 97 (.xls) file. If EIA returns an HTML error page
# instead of a file, this check catches it before it is saved.
XLS_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@dataclass(frozen=True)
class Series:
    series_id: str    # EIA code
    name: str         # metric name used in the project (CSV file and column name)
    description: str
    units: str
    freq: str         # "d" daily, "w" weekly
    group: str        # "prices" or "weekly"

    @property
    def url(self) -> str:
        return EIA_HIST_URL.format(series_id=self.series_id, freq=self.freq)

    @property
    def page_url(self) -> str:
        """Human-readable EIA page for the series (for the README / paper)."""
        return EIA_PAGE_URL.format(series_id=self.series_id, freq=self.freq.upper())

    @property
    def xls_path(self) -> Path:
        """Where the original EIA download is kept (not committed)."""
        return RAW_XLS_DIR / f"{self.series_id}{self.freq}.xls"

    @property
    def csv_path(self) -> Path:
        """The CSV used by the rest of the project."""
        folder = RAW_PRICES_DIR if self.group == "prices" else RAW_WEEKLY_DIR
        return folder / f"{self.name}.csv"


SERIES: list[Series] = [
    # Daily spot prices (target: weekly realised volatility)
    Series("RWTC", "wti", "WTI spot price, Cushing OK (FOB)", "USD per barrel", "d", "prices"),
    Series("RBRTE", "brent", "Brent spot price, Europe (FOB)", "USD per barrel", "d", "prices"),
    # Weekly Petroleum Status Report (features)
    Series("WCESTUS1", "crude_stocks", "U.S. commercial crude oil stocks, excl. SPR",
           "thousand barrels", "w", "weekly"),
    Series("W_EPC0_SAX_YCUOK_MBBL", "cushing_stocks", "Cushing OK crude oil stocks",
           "thousand barrels", "w", "weekly"),
    Series("WCSSTUS1", "spr_stocks", "Strategic Petroleum Reserve crude oil stocks",
           "thousand barrels", "w", "weekly"),
    Series("WGTSTUS1", "gasoline_stocks", "U.S. total motor gasoline stocks",
           "thousand barrels", "w", "weekly"),
    Series("WDISTUS1", "distillate_stocks", "U.S. distillate fuel oil stocks",
           "thousand barrels", "w", "weekly"),
    Series("WCRFPUS2", "crude_production", "U.S. field production of crude oil",
           "thousand barrels per day", "w", "weekly"),
    Series("WCRRIUS2", "refinery_input", "U.S. refiner net input of crude oil",
           "thousand barrels per day", "w", "weekly"),
    Series("WPULEUS3", "refinery_utilization", "U.S. refinery utilisation (% of operable capacity)",
           "percent", "w", "weekly"),
    Series("WCEIMUS2", "crude_imports", "U.S. commercial crude oil imports, excl. SPR",
           "thousand barrels per day", "w", "weekly"),
    Series("WCREXUS2", "crude_exports", "U.S. crude oil exports",
           "thousand barrels per day", "w", "weekly"),
    Series("WRPUPUS2", "product_supplied", "U.S. product supplied of petroleum products (demand proxy)",
           "thousand barrels per day", "w", "weekly"),
]

SERIES_BY_ID = {s.series_id: s for s in SERIES}
SERIES_BY_NAME = {s.name: s for s in SERIES}


def series_table(series: list[Series] = SERIES) -> pd.DataFrame:
    """The list of series as a table (for display and for the paper)."""
    return pd.DataFrame(
        [
            {"series_id": s.series_id, "metric": s.name, "description": s.description,
             "units": s.units, "frequency": "daily" if s.freq == "d" else "weekly",
             "csv_file": s.csv_path.relative_to(ROOT).as_posix(), "url": s.url}
            for s in series
        ]
    )


# ---------------------------------------------------------------------------
# 1. Download the original .xls files
# ---------------------------------------------------------------------------

def download_series(s: Series, force: bool = False, timeout: int = 60) -> str:
    """Download one EIA history file into data/raw/xls/. Returns a status string.

    An existing file is kept unless force=True, so the notebook can be re-run
    quickly and files you copied in by hand are not overwritten.
    """
    if s.xls_path.exists() and not force:
        return "already present"

    s.xls_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        s.url, timeout=timeout, headers={"User-Agent": "ada-oil-volatility student project"}
    )
    response.raise_for_status()
    if not response.content.startswith(XLS_SIGNATURE):
        raise ValueError(f"{s.url} did not return an .xls file (check the series code)")

    # Write to a temporary file first, so a failed download never leaves a broken file
    tmp_path = s.xls_path.with_suffix(".part")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(s.xls_path)
    return "downloaded"


def download_all(series: list[Series] = SERIES, force: bool = False,
                 pause: float = 1.0) -> pd.DataFrame:
    """Download every series; one failure does not stop the others."""
    rows = []
    for s in series:
        try:
            status = download_series(s, force=force)
        except Exception as exc:  # report the problem and carry on
            status = f"FAILED: {exc}"
        rows.append({"series_id": s.series_id, "status": status, "file": s.xls_path.name})
        if status == "downloaded":
            time.sleep(pause)  # be polite to the EIA server
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Read an .xls file and convert a whole folder to named CSV files
# ---------------------------------------------------------------------------

def read_eia_xls(path: Path | str) -> pd.Series:
    """Read one EIA history file and return a date-indexed series.

    The data sheet ("Data 1") has a few title rows, then a header row whose
    first cell is "Date", then one row per observation. The row "Sourcekey"
    holds the EIA series code, which is used as the series name, so a file is
    recognised even if it was renamed (e.g. "RWTCd (1).xls").
    """
    raw = pd.read_excel(path, sheet_name="Data 1", header=None, engine="xlrd")
    first_col = raw.iloc[:, 0].astype(str).str.strip()

    header_rows = first_col.index[first_col == "Date"]
    if len(header_rows) == 0:
        raise ValueError(f"{path}: no 'Date' header row found in sheet 'Data 1'")
    header_row = header_rows[0]

    key_rows = first_col.index[first_col == "Sourcekey"]
    series_id = str(raw.iloc[key_rows[0], 1]).strip() if len(key_rows) else Path(path).stem

    data = raw.iloc[header_row + 1:, :2].copy()
    data.columns = ["date", "value"]
    if pd.api.types.is_numeric_dtype(data["date"]):
        # Dates stored as Excel serial numbers
        data["date"] = pd.to_datetime(data["date"], unit="D", origin="1899-12-30")
    else:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")

    if data["date"].duplicated().any():
        raise ValueError(f"{path}: duplicated dates")

    return data.set_index("date")["value"].rename(series_id)


def convert_xls_to_csv(src_dir: Path | str = RAW_XLS_DIR) -> pd.DataFrame:
    """Convert every EIA .xls file in src_dir to data/raw/<group>/<metric>.csv.

    Each file is identified from the series code stored inside it, not from its
    file name. Files that are not EIA history files, or are not used in the
    project, are skipped and reported. The original .xls files are not changed.
    """
    src_dir = Path(src_dir).expanduser()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {src_dir}")

    rows, done = [], {}
    for path in sorted(src_dir.glob("*.xls")):
        row = {"file": path.name}
        try:
            values = read_eia_xls(path)
        except Exception as exc:
            rows.append({**row, "status": f"skipped: not an EIA history file ({exc})"})
            continue
        s = SERIES_BY_ID.get(values.name)
        if s is None:
            rows.append({**row, "series_id": values.name,
                         "status": "skipped: series not used in the project"})
            continue
        if s.series_id in done:
            rows.append({**row, "series_id": s.series_id,
                         "status": f"skipped: duplicate of {done[s.series_id]}"})
            continue

        s.csv_path.parent.mkdir(parents=True, exist_ok=True)
        values.rename(s.name).rename_axis("date").to_csv(s.csv_path, date_format="%Y-%m-%d")
        done[s.series_id] = path.name
        rows.append({**row, "series_id": s.series_id,
                     "saved_as": s.csv_path.relative_to(ROOT).as_posix(), "status": "converted"})

    return pd.DataFrame(rows, columns=["file", "series_id", "saved_as", "status"])


# ---------------------------------------------------------------------------
# 3. Read the CSV files (used by the later stages)
# ---------------------------------------------------------------------------

def read_series(name: str) -> pd.Series:
    """Read one metric from its CSV, e.g. read_series("crude_stocks")."""
    s = SERIES_BY_NAME[name]
    data = pd.read_csv(s.csv_path, parse_dates=["date"], index_col="date")
    return data[name]


def build_manifest(series: list[Series] = SERIES, save: bool = True) -> pd.DataFrame:
    """Check every CSV and record what it contains (data/raw/manifest.csv).

    The manifest documents the data set for the paper: coverage and number of
    observations of each file. The last date shows which EIA release was used.
    """
    rows = []
    for s in series:
        row = {"series_id": s.series_id, "metric": s.name, "description": s.description,
               "units": s.units, "frequency": "daily" if s.freq == "d" else "weekly",
               "file": s.csv_path.relative_to(ROOT).as_posix(), "url": s.url,
               "committed_to_git": s.group == "weekly"}
        if not s.csv_path.exists():
            rows.append({**row, "status": "missing"})
            continue
        try:
            values = read_series(s.name)
        except Exception as exc:
            rows.append({**row, "status": f"unreadable: {exc}"})
            continue
        rows.append({**row, "status": "ok",
                     "first_date": values.index.min().date(),
                     "last_date": values.index.max().date(),
                     "n_obs": int(values.notna().sum()),
                     "n_missing": int(values.isna().sum())})

    manifest = pd.DataFrame(rows)
    if save:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert EIA .xls files to one CSV per metric in data/raw/."
    )
    parser.add_argument("folder", nargs="?", default=str(RAW_XLS_DIR),
                        help="folder containing the downloaded .xls files "
                             "(default: data/raw/xls)")
    args = parser.parse_args()

    result = convert_xls_to_csv(args.folder)
    print(result.fillna("").to_string(index=False))

    manifest = build_manifest()
    missing = manifest.loc[manifest["status"] != "ok", "metric"].tolist()
    print(f"\n{(manifest['status'] == 'ok').sum()} of {len(manifest)} metrics available as CSV.")
    if missing:
        print("Still missing:", ", ".join(missing))
