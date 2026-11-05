"""Download the EIA data, convert it to CSV and read it back.

EIA publishes the full history of each series as an Excel 97 (.xls) file at
https://www.eia.gov/dnav/pet/hist_xls/<SERIES_ID><frequency>.xls
(frequency: "d" = daily, "w" = weekly). No account or API key is needed.

Those .xls files are used for the 11 weekly series, but NOT for the daily
prices: the published price files silently skip hundreds of ordinary trading
days (about 757 for WTI and 900 for Brent). The daily prices therefore come
from the EIA API v2, which returns the complete series and identical prices on
the days both sources cover. The API needs a free key, read from the
environment variable EIA_API_KEY (see the README).

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

from .config import (EIA_API_KEY_VAR, ENV_PATH, MANIFEST_PATH, RAW_PRICES_DIR,
                     RAW_WEEKLY_DIR, RAW_XLS_DIR, ROOT)

EIA_HIST_URL = "https://www.eia.gov/dnav/pet/hist_xls/{series_id}{freq}.xls"

# EIA API v2: the spot price route, used for the daily prices
EIA_API_ROUTE = "https://api.eia.gov/v2/petroleum/pri/spt"
EIA_API_PAGE_SIZE = 5000   # the API returns at most 5000 rows per request
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
# 1b. Download the daily prices from the EIA API v2
# ---------------------------------------------------------------------------

def eia_api_key() -> str:
    """Read the EIA API key from the environment (loading .env if needed).

    The key is a secret: it is kept in .env, which is in .gitignore, and is
    never written to a notebook, a CSV or the README. This function returns it
    for use in a request and nothing else ever prints it.
    """
    import os

    if not os.environ.get(EIA_API_KEY_VAR):
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover
            raise RuntimeError(
                "python-dotenv is not installed: run `pip install -r requirements.txt`"
            ) from None
        load_dotenv(ENV_PATH)

    key = os.environ.get(EIA_API_KEY_VAR)
    if not key:
        raise RuntimeError(
            f"No EIA API key found. Create a file called .env in the repository root "
            f"containing one line:\n    {EIA_API_KEY_VAR}=your_key_here\n"
            "A free key takes a minute to obtain at "
            "https://www.eia.gov/opendata/register.php"
        )
    return key


def _api_get(path: str, params: dict, timeout: int = 120) -> dict:
    """One request to the EIA API, with the key kept out of every message.

    `requests.raise_for_status()` and the exceptions raised by `requests` put
    the full URL — and therefore the key — into their message. So the status is
    checked by hand and every error is re-raised with a clean message, so the
    key can never reach a traceback, a log or a notebook output.
    """
    url = f"{EIA_API_ROUTE}{path}"
    try:
        response = requests.get(
            url, params={**params, "api_key": eia_api_key()}, timeout=timeout,
            headers={"User-Agent": "ada-oil-volatility student project"},
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"the EIA API could not be reached ({type(exc).__name__}); check the connection"
        ) from None

    if response.status_code == 403:
        raise RuntimeError(
            "the EIA API refused the key (HTTP 403). Check the value of "
            f"{EIA_API_KEY_VAR} in .env; a free key is available at "
            "https://www.eia.gov/opendata/register.php"
        ) from None
    if response.status_code != 200:
        raise RuntimeError(
            f"the EIA API returned HTTP {response.status_code} for {path}"
        ) from None

    return response.json()["response"]


def api_route_metadata() -> dict:
    """The spot price route's metadata: frequencies, facets and period range.

    Used by notebook 01 to confirm the facet name (`series`) and the codes
    (RWTC, RBRTE) before downloading, and as the small request that checks the
    key works.
    """
    return _api_get("/", {})


def api_series_names() -> pd.DataFrame:
    """The series available on the spot price route, with their descriptions."""
    facets = _api_get("/facet/series", {})["facets"]
    return pd.DataFrame(facets).rename(columns={"id": "series_id", "name": "description"})


def download_price_series(s: Series) -> pd.Series:
    """Download one daily price series in full from the EIA API.

    The API returns at most 5000 rows per request, so the pages are requested
    one after another with an increasing `offset` until a short page arrives.
    Rows are asked for in ascending date order, so the pages join up directly.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page = _api_get("/data/", {
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": s.series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": EIA_API_PAGE_SIZE,
        })
        data = page["data"]
        rows.extend(data)
        if len(data) < EIA_API_PAGE_SIZE:
            break
        offset += EIA_API_PAGE_SIZE

    values = pd.Series(
        {pd.Timestamp(row["period"]): row["value"] for row in rows},
        dtype="float64", name=s.name,
    ).sort_index()
    if values.index.has_duplicates:
        raise ValueError(f"{s.series_id}: the API returned duplicated dates")
    return values.rename_axis("date")


def read_price_xls(s: Series) -> pd.Series | None:
    """The same price series from the downloaded .xls file, or None if absent.

    Used only to check the API against the published file and to fill the few
    days the API happens to be missing.
    """
    if not s.xls_path.exists():
        return None
    return read_eia_xls(s.xls_path).rename(s.name)


def download_prices(names: list[str] | None = None, fill_from_xls: bool = True) -> pd.DataFrame:
    """Write data/raw/prices/<name>.csv from the API, and report what changed.

    The API is the main source because the .xls price files skip hundreds of
    trading days. The .xls file is still used for two things:

    * a check that the two sources agree — the prices must be identical on
      every date they share;
    * filling the handful of days that are in the .xls but not in the API
      (8 for WTI, 9 for Brent), so neither source's gaps survive. Set
      `fill_from_xls=False` to use the API alone.

    The CSV keeps the format of stage 1: two columns, `date` and `<name>`.
    """
    series = [s for s in SERIES if s.group == "prices"
              and (names is None or s.name in names)]
    rows = []
    for s in series:
        api = download_price_series(s)
        old = read_price_xls(s) if fill_from_xls else None

        row = {"metric": s.name, "series_id": s.series_id, "n_api": len(api)}
        if old is None:
            row["n_xls"] = None
            row["n_differences"] = None
            row["added_by_api"] = None
            row["filled_from_xls"] = 0
            prices = api
        else:
            common = api.index.intersection(old.index)
            differences = (api.loc[common] - old.loc[common]).abs() > 1e-9
            missing_from_api = old.index.difference(api.index)
            # API first, then the days only the .xls has
            prices = pd.concat([api, old.loc[missing_from_api]]).sort_index()
            row.update({
                "n_xls": len(old),
                "n_differences": int(differences.sum()),
                "added_by_api": len(api.index.difference(old.index)),
                "filled_from_xls": len(missing_from_api),
            })

        s.csv_path.parent.mkdir(parents=True, exist_ok=True)
        prices.rename(s.name).rename_axis("date").to_csv(s.csv_path, date_format="%Y-%m-%d")
        row["n_saved"] = len(prices)
        row["saved_as"] = s.csv_path.relative_to(ROOT).as_posix()
        rows.append(row)

    return pd.DataFrame(rows)


def observations_per_year(name: str) -> pd.Series:
    """Number of observations per calendar year for one metric, from its CSV."""
    values = read_series(name).dropna()
    return values.groupby(values.index.year).size().rename(name).rename_axis("year")


def compare_price_sources(name: str) -> pd.DataFrame:
    """Compare the API prices with the published .xls file, year by year.

    Returns one row per year with: how many days each source has, how many days
    the API adds, how many days only the .xls has, and on how many shared dates
    the two prices disagree. Used by notebook 01 to show that the API fixes the
    gaps without changing any price. Returns an empty frame if the .xls file has
    not been downloaded.
    """
    s = SERIES_BY_NAME[name]
    old = read_price_xls(s)
    if old is None:
        return pd.DataFrame()
    api = download_price_series(s)

    common = api.index.intersection(old.index)
    differs = pd.Series((api.loc[common] - old.loc[common]).abs() > 1e-9, index=common)
    added = api.index.difference(old.index)
    only_xls = old.index.difference(api.index)

    years = sorted({*api.index.year, *old.index.year})
    table = pd.DataFrame({
        "n_api": pd.Series(api.index.year).value_counts().reindex(years, fill_value=0),
        "n_xls": pd.Series(old.index.year).value_counts().reindex(years, fill_value=0),
        "added_by_api": pd.Series(added.year).value_counts().reindex(years, fill_value=0),
        "only_in_xls": pd.Series(only_xls.year).value_counts().reindex(years, fill_value=0),
        "price_differs": differs.groupby(common.year).sum().reindex(years, fill_value=0),
    })
    return table.rename_axis("year").astype(int)


def weekly_observations_per_year() -> pd.DataFrame:
    """Observations per year for the 11 weekly EIA series (expect 52 or 53)."""
    columns = {s.name: observations_per_year(s.name)
               for s in SERIES if s.group == "weekly"}
    return pd.DataFrame(columns).rename_axis("year")


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


def convert_xls_to_csv(src_dir: Path | str = RAW_XLS_DIR,
                       groups: tuple[str, ...] = ("weekly",)) -> pd.DataFrame:
    """Convert every EIA .xls file in src_dir to data/raw/<group>/<metric>.csv.

    Each file is identified from the series code stored inside it, not from its
    file name. Files that are not EIA history files, or are not used in the
    project, are skipped and reported. The original .xls files are not changed.

    Only the groups listed in `groups` are written. By default the prices are
    left out: the published price .xls files skip hundreds of trading days, so
    `download_prices()` writes those CSVs from the EIA API instead. The price
    .xls files are still read, for the comparison in notebook 01.
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
        if s.group not in groups:
            rows.append({**row, "series_id": s.series_id,
                         "status": "skipped: written by download_prices() from the EIA API"})
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
               "file": s.csv_path.relative_to(ROOT).as_posix(),
               "source": "EIA API v2" if s.group == "prices" else "EIA .xls history file",
               "url": EIA_API_ROUTE if s.group == "prices" else s.url,
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
    print("\nThe daily prices are not converted from .xls: run notebook 01, or\n"
          "`python -c \"from src.data_utils import download_prices; print(download_prices())\"`,\n"
          "to download them from the EIA API (needs EIA_API_KEY in .env).")

    manifest = build_manifest()
    missing = manifest.loc[manifest["status"] != "ok", "metric"].tolist()
    print(f"\n{(manifest['status'] == 'ok').sum()} of {len(manifest)} metrics available as CSV.")
    if missing:
        print("Still missing:", ", ".join(missing))
