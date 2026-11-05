"""Stage 2: clean the data and build the weekly panel.

The panel has one row per week ending Friday. Every value in a row is known at
the close of that Friday, so the later stages cannot accidentally use the
future:

* price variables are computed from the daily prices of that same week;
* EIA variables are the latest ones *published* on or before that Friday.

An EIA weekly value refers to the week ending Friday but is only released the
following Wednesday (Thursday after a US federal holiday). We use a
conservative release date of week-ending + 6 days (the Thursday), so the panel
week t can only ever see the EIA week ending t - 7 days or earlier.

Two details of the daily prices are handled explicitly:

* WTI settled at -36.98 USD on 2020-04-20, so ln(P) does not exist. Instead of
  losing two returns, the non-positive price is skipped and ONE return is
  measured from the last positive price to the next positive price. That return
  covers two trading days, which is recorded in `days_covered`.
* a week does not always contain five trading days (holidays, and the few days
  still missing from the price series). Because realised variance is a SUM of
  squared returns, a four-day week is mechanically smaller than a five-day week.
  `rv_adj = rv * 5 / n_days` puts every week on the same five-day scale, and is
  the quantity the models predict.

Written with the help of Claude (Anthropic), which was asked to produce the
cleaning, weekly aggregation and release-date alignment code for this project
(declared in the paper appendix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SAMPLE_END, SAMPLE_START, TRAIN_END, VALID_END
from .data_utils import SERIES, read_series

# Weeks end on Friday, like the EIA weeks.
WEEK_FREQ = "W-FRI"

# Weeks per year, used to annualise the weekly realised variance.
WEEKS_PER_YEAR = 52

# A full trading week. Realised variance is rescaled to this many days.
DAYS_PER_WEEK = 5

# A week needs at least this many trading days covered for its realised
# variance to be used.
MIN_DAYS_PER_WEEK = 3

# The price series of the project; WTI is the main one, Brent the robustness check.
PRICE_NAMES = ["wti", "brent"]

# The 11 EIA weekly series, in the order used in the panel.
EIA_NAMES = [s.name for s in SERIES if s.group == "weekly"]

# The price variables computed for each price series, in panel order.
PRICE_VARS = ["rv", "rv_adj", "n_days", "log_rv", "rvol_ann", "ret_w", "rv_fss"]


# ---------------------------------------------------------------------------
# A. Daily returns
# ---------------------------------------------------------------------------

def daily_log_returns(prices: pd.Series) -> pd.DataFrame:
    """Daily log returns, bridging over any price that is not positive.

    Returns a frame indexed by date with two columns:

    * `ret`           ln(P_t / P_last_positive), where P_t is a positive price;
    * `days_covered`  how many trading days that return spans (normally 1).

    The logarithm of a non-positive price does not exist. Rather than drop the
    return into it and the return out of it, the non-positive observation is
    skipped and a single return is measured from the last positive price to the
    next positive price. It is dated on the next positive price and marked as
    covering 2 trading days.

    For WTI this bridges 2020-04-17 (18.31) to 2020-04-21 (8.91) across the
    negative settlement of 2020-04-20. Nothing is filled or interpolated: the
    bridged return is the true price change over those two days, and the fact
    that it took two days instead of one is kept in `days_covered`.
    """
    prices = prices.sort_index()
    positive = prices[prices > 0]

    # Position of each positive price in the full trading calendar of the
    # series, so a skipped day is counted rather than ignored.
    position = pd.Series(np.arange(len(prices)), index=prices.index)
    days_covered = position.loc[positive.index].diff()

    returns = np.log(positive).diff()
    return pd.DataFrame({"ret": returns, "days_covered": days_covered}).rename_axis("date")


def bridged_returns(name: str = "wti") -> pd.DataFrame:
    """The returns that span more than one trading day, with the prices used.

    Normally empty. For WTI it holds the single return that bridges the
    negative price of 2020-04-20, so the notebook can show exactly which prices
    were used and the paper can quote them.
    """
    prices = read_series(name).sort_index()
    frame = daily_log_returns(prices)
    bridged = frame[frame["days_covered"] > 1].copy()
    if bridged.empty:
        return bridged.assign(from_date=pd.NaT, from_price=np.nan, price=np.nan,
                              skipped_dates="", skipped_prices="")

    positive_dates = prices[prices > 0].index
    rows = []
    for date in bridged.index:
        previous = positive_dates[positive_dates < date][-1]
        skipped = prices.loc[previous:date].iloc[1:-1]
        rows.append({
            "from_date": previous,
            "from_price": prices.loc[previous],
            "price": prices.loc[date],
            "skipped_dates": ", ".join(d.strftime("%Y-%m-%d") for d in skipped.index),
            "skipped_prices": ", ".join(f"{v:g}" for v in skipped.to_numpy()),
        })
    return bridged.join(pd.DataFrame(rows, index=bridged.index))


def build_daily_returns(names: list[str] = PRICE_NAMES) -> pd.DataFrame:
    """Daily returns of every price series, as `<name>_ret` and `<name>_days`.

    Each series keeps its own trading days: the returns are computed before the
    series are put side by side, so a day missing from one price series does not
    affect the other.
    """
    columns = {}
    for name in names:
        frame = daily_log_returns(read_series(name))
        columns[f"{name}_ret"] = frame["ret"]
        columns[f"{name}_days"] = frame["days_covered"]
    return pd.DataFrame(columns).sort_index().rename_axis("date")


def non_positive_prices(name: str) -> pd.Series:
    """The prices that are zero or negative (WTI: one, on 2020-04-20)."""
    prices = read_series(name)
    return prices[prices <= 0]


# ---------------------------------------------------------------------------
# B. Weekly price variables
# ---------------------------------------------------------------------------

def _fss_sum(week_returns: pd.Series) -> float:
    """French-Schwert-Stambaugh variance of one week, before rescaling.

    `sum(r^2) + 2 * sum(r_t * r_{t-1})`. The second term adds the sample
    autocovariance of consecutive returns, which corrects the realised variance
    when daily returns are autocorrelated (French, Schwert & Stambaugh, 1987).
    Only pairs of returns that both fall inside the week are used, so no
    information crosses a week boundary.

    The correction can make the result zero or negative when returns are
    strongly negatively autocorrelated; the caller drops those weeks.
    """
    values = week_returns.dropna().to_numpy()
    if values.size == 0:
        return np.nan
    squares = float((values ** 2).sum())
    cross = float((values[1:] * values[:-1]).sum()) if values.size > 1 else 0.0
    return squares + 2 * cross


def weekly_price_vars(prices: pd.Series, returns: pd.DataFrame,
                      min_days: int = MIN_DAYS_PER_WEEK) -> pd.DataFrame:
    """Weekly realised volatility and weekly return for one price series.

    Weeks end on Friday ("W-FRI"), so a week runs from Saturday to Friday and
    is labelled with its Friday, exactly like an EIA week. Columns:

    * `rv`       realised variance: the sum of the squared daily log returns;
    * `n_days`   trading days covered by those returns (a bridged return counts
                 for the days it spans, so a normal full week gives 5);
    * `rv_adj`   `rv * 5 / n_days`: the variance put on a five-day scale, so
                 that a holiday week is comparable with a full week;
    * `log_rv`   `ln(rv_adj)`, the quantity the models predict (realised
                 variance is strongly right-skewed, its log is close to normal);
    * `rvol_ann` `100 * sqrt(52 * rv_adj)`, annualised volatility in percent,
                 used for figures and tables only;
    * `ret_w`    weekly log return: last price of the week against the last
                 price of the previous week;
    * `rv_fss`   robustness proxy: the same variance with the French-Schwert-
                 Stambaugh autocovariance correction, also on a five-day scale.

    `rv`, `rv_adj`, `log_rv` and `rvol_ann` are set to NaN when fewer than
    `min_days` trading days are covered, or when `rv` is exactly 0 (the log
    would be undefined). `n_days` is always kept, so the reason stays visible.
    """
    weekly = returns.resample(WEEK_FREQ)
    # min_count=1 keeps a week with no valid return as NaN instead of 0.0
    rv = weekly["ret"].apply(lambda r: (r ** 2).sum(min_count=1))
    n_days = weekly["days_covered"].sum(min_count=1).fillna(0)

    usable = (n_days >= min_days) & (rv > 0)
    rv = rv.where(usable)
    rv_adj = rv * DAYS_PER_WEEK / n_days.where(n_days > 0)
    log_rv = np.log(rv_adj)
    rvol_ann = 100 * np.sqrt(WEEKS_PER_YEAR * rv_adj)

    # Weekly return from the last price of each week. The log needs a positive
    # price, which is why the week ending 2020-04-24 is checked, not assumed.
    week_close = prices.resample(WEEK_FREQ).last()
    positive = week_close > 0
    ret_w = np.log(week_close.where(positive)).diff()
    ret_w = ret_w.where(positive & positive.shift(1))

    # Robustness proxy: the same variance with the FSS autocovariance correction,
    # on the same five-day scale. Dropped when the correction makes it non-positive.
    rv_fss = weekly["ret"].apply(_fss_sum) * DAYS_PER_WEEK / n_days.where(n_days > 0)
    rv_fss = rv_fss.where((n_days >= min_days) & (rv_fss > 0))

    return pd.DataFrame(
        {"rv": rv, "rv_adj": rv_adj, "n_days": n_days,
         "log_rv": log_rv, "rvol_ann": rvol_ann, "ret_w": ret_w, "rv_fss": rv_fss}
    ).rename_axis("date")


def thin_weeks(weekly: pd.DataFrame, min_days: int = MIN_DAYS_PER_WEEK) -> pd.DataFrame:
    """The weeks whose realised variance was dropped for covering too few days."""
    thin = weekly["n_days"] < min_days
    return weekly.loc[thin, ["n_days", "ret_w"]]


def build_weekly_price_panel(returns: pd.DataFrame,
                             names: list[str] = PRICE_NAMES,
                             min_days: int = MIN_DAYS_PER_WEEK) -> pd.DataFrame:
    """The weekly price variables of every price series, as `<name>_<variable>`.

    Takes the daily returns computed by `build_daily_returns` so that the
    returns used in the panel are exactly the ones saved to
    `data/processed/daily_returns.csv`.
    """
    frames = []
    for name in names:
        daily = returns[[f"{name}_ret", f"{name}_days"]].rename(
            columns={f"{name}_ret": "ret", f"{name}_days": "days_covered"})
        weekly = weekly_price_vars(read_series(name), daily, min_days=min_days)
        frames.append(weekly.add_prefix(f"{name}_"))
    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# C. EIA data aligned to its release dates
# ---------------------------------------------------------------------------

def build_eia_table(names: list[str] = EIA_NAMES) -> pd.DataFrame:
    """The 11 EIA weekly series side by side, one row per EIA week.

    The original units and names are kept: stage 2 only aligns the data in
    time. A series that does not cover a week is NaN there, which is why
    `cushing_stocks` is empty before 2004.
    """
    table = pd.concat({name: read_series(name) for name in names}, axis=1)
    return table[names].sort_index().rename_axis("eia_week")


def non_friday_dates(table: pd.DataFrame) -> pd.DatetimeIndex:
    """The EIA week-ending dates that are not a Friday (expected: none)."""
    index = pd.DatetimeIndex(table.index)
    return index[index.dayofweek != 4]  # Monday = 0, so Friday = 4


def add_release_dates(table: pd.DataFrame, lag_days: int = 6) -> pd.DataFrame:
    """Add the date on which each EIA week became public.

    A week ending Friday is published the following Wednesday at 10:30 ET, or
    the Thursday when a US federal holiday falls in between. We always use the
    Thursday (week-ending + 6 days) instead of looking up the holiday calendar:
    it is one day late at worst, never too early, so the panel can never use a
    number before it existed.
    """
    out = table.reset_index()
    out["release_date"] = out["eia_week"] + pd.Timedelta(days=lag_days)
    return out


def align_eia_to_weeks(week_index: pd.DatetimeIndex,
                       table: pd.DataFrame,
                       lag_days: int = 6) -> pd.DataFrame:
    """Attach to each panel week the latest EIA data published by that Friday.

    For a panel week ending on Friday t, `pd.merge_asof(direction="backward")`
    picks the last EIA row whose `release_date` is on or before t. Because the
    release date is the week-ending date plus 6 days, that row is normally the
    EIA week ending t - 7 days: the week before the panel week, published on the
    Wednesday inside the panel week. The `eia_week` column records which EIA
    week each row actually used.
    """
    panel_weeks = pd.DataFrame({"date": pd.DatetimeIndex(week_index)})
    releases = add_release_dates(table, lag_days=lag_days).sort_values("release_date")

    merged = pd.merge_asof(
        panel_weeks, releases,
        left_on="date", right_on="release_date",
        direction="backward",
    )
    return merged.set_index("date").drop(columns="release_date").rename_axis("date")


def lookahead_violations(panel: pd.DataFrame, min_lag_days: int = 7) -> pd.DataFrame:
    """Rows where the EIA week used is too recent (expected: none).

    Each row must use an EIA week ending at least `min_lag_days` before the
    panel week, otherwise the panel would contain data published after that
    Friday.
    """
    used = panel["eia_week"]
    gap = pd.Series(panel.index, index=panel.index) - used
    bad = used.notna() & (gap < pd.Timedelta(days=min_lag_days))
    return pd.DataFrame({"eia_week": used, "gap_days": gap.dt.days}).loc[bad]


# ---------------------------------------------------------------------------
# D. The weekly panel
# ---------------------------------------------------------------------------

def week_index(start: str = SAMPLE_START, end: str = SAMPLE_END) -> pd.DatetimeIndex:
    """Every week-ending Friday of the sample period, from config.py."""
    return pd.date_range(start=start, end=end, freq=WEEK_FREQ)


def negative_price_weeks(week_index: pd.DatetimeIndex, name: str = "wti") -> pd.Series:
    """1 for a week whose daily prices include a non-positive value, else 0.

    A flag for robustness checks only — it marks the week ending 2020-04-24,
    whose variance is built across the negative WTI settlement. It must never be
    used as a model input: it is known only after the week it describes.
    """
    prices = read_series(name)
    weeks = prices[prices <= 0].index.to_series().resample(WEEK_FREQ).count()
    flag = weeks.reindex(week_index, fill_value=0)
    return (flag > 0).astype("int64").rename(f"{name}_negative_price_week")


def build_weekly_panel(start: str = SAMPLE_START, end: str = SAMPLE_END,
                       min_days: int = MIN_DAYS_PER_WEEK,
                       lag_days: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the whole stage 2 output: the daily returns and the weekly panel.

    The columns are, in order: the six price variables of WTI, the same six for
    Brent, the negative-price flag, `eia_week`, and the 11 EIA series in their
    original units. No row and no column is dropped, so weeks without usable
    data stay in the table as NaN and the gaps remain visible in the coverage
    table.
    """
    returns = build_daily_returns()
    weeks = week_index(start, end)

    prices_panel = build_weekly_price_panel(returns, min_days=min_days).reindex(weeks)
    eia_panel = align_eia_to_weeks(weeks, build_eia_table(), lag_days=lag_days)
    flag = negative_price_weeks(weeks)

    panel = pd.concat([prices_panel, flag, eia_panel], axis=1)
    ordered = ([f"{name}_{var}" for name in PRICE_NAMES for var in PRICE_VARS]
               + [flag.name, "eia_week"] + EIA_NAMES)
    return returns, panel[ordered].rename_axis("date")


# ---------------------------------------------------------------------------
# E. Checks
# ---------------------------------------------------------------------------

def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """First and last non-missing week, and the counts, for every column."""
    rows = []
    for column in panel.columns:
        values = panel[column]
        present = values.dropna()
        rows.append({
            "column": column,
            "first_week": present.index.min().date() if len(present) else None,
            "last_week": present.index.max().date() if len(present) else None,
            "n_obs": int(len(present)),
            "n_missing": int(values.isna().sum()),
            "pct_missing": round(100 * values.isna().mean(), 2),
        })
    return pd.DataFrame(rows)


def price_completeness(panel: pd.DataFrame, names: list[str] = PRICE_NAMES,
                       min_days: int = MIN_DAYS_PER_WEEK) -> pd.DataFrame:
    """How many trading days each week covers, year by year.

    `n_days` is the number of trading days behind each weekly realised
    variance; a full week has 5, and a week with a US holiday has 4. Since the
    prices now come from the EIA API, weeks below 5 should be holiday weeks
    only. `rv_dropped` counts the weeks whose variance was removed by the
    `n_days < min_days` rule.
    """
    rows = []
    years = panel.index.year
    for name in names:
        n_days = panel[f"{name}_n_days"]
        rows.append(pd.DataFrame({
            (name, "mean_n_days"): n_days.groupby(years).mean().round(2),
            (name, "weeks_below_5"): n_days.lt(5).groupby(years).sum(),
            (name, "rv_dropped"): panel[f"{name}_rv_adj"].isna().groupby(years).sum(),
        }))
    out = pd.concat(rows, axis=1)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out.rename_axis("year")


def check_panel(panel: pd.DataFrame, min_lag_days: int = 7) -> None:
    """Raise if the panel breaks a rule the later stages rely on.

    Checks that the weekly index is unique, sorted and made only of Fridays,
    and that no row uses EIA data published after its own Friday.
    """
    index = pd.DatetimeIndex(panel.index)
    if not index.is_unique:
        raise AssertionError("the weekly index has duplicates")
    if not index.is_monotonic_increasing:
        raise AssertionError("the weekly index is not sorted")
    not_friday = index[index.dayofweek != 4]
    if len(not_friday):
        raise AssertionError(f"these panel weeks do not end on a Friday: {list(not_friday)}")

    violations = lookahead_violations(panel, min_lag_days=min_lag_days)
    if not violations.empty:
        raise AssertionError(
            f"{len(violations)} rows use EIA data published after their own week:\n"
            f"{violations.head().to_string()}"
        )


# ---------------------------------------------------------------------------
# E2. Checks on the assumptions behind the proxy
# ---------------------------------------------------------------------------

# The chronological sub-periods of the project, from config.py.
SUB_PERIODS: dict[str, tuple[str, str]] = {
    "train 2000-2014": (SAMPLE_START, TRAIN_END),
    "tune 2015-2018": (TRAIN_END, VALID_END),
    "test 2019-2026": (VALID_END, SAMPLE_END),
    "whole sample": (SAMPLE_START, SAMPLE_END),
}


def return_diagnostics(returns: pd.DataFrame, names: list[str] = PRICE_NAMES,
                       periods: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Check the two assumptions behind using realised variance as a proxy.

    Realised variance `sum(r^2)` estimates the variance of the daily returns
    only if (i) their mean is negligible compared with their standard
    deviation, so that not subtracting the mean costs nothing, and (ii) the
    returns are close to serially uncorrelated, so that the squared terms alone
    capture the variation (Andersen et al., 2003). When (ii) fails, the
    French-Schwert-Stambaugh correction in `rv_fss` is the remedy.

    One row per series and sub-period, with:

    * `mean`, `std` and `mean_over_std` for assumption (i);
    * `acf1`, the lag-1 autocorrelation, and `band_95` = 1.96/sqrt(n), the
      width of the band inside which `acf1` is indistinguishable from zero,
      for assumption (ii).
    """
    periods = SUB_PERIODS if periods is None else periods
    rows = []
    for name in names:
        series = returns[f"{name}_ret"].dropna()
        for label, (start, end) in periods.items():
            window = series.loc[start:end]
            n = len(window)
            mean, std = window.mean(), window.std()
            acf1 = window.autocorr(lag=1)
            band = 1.96 / np.sqrt(n) if n else np.nan
            rows.append({
                "series": name, "period": label, "n": n,
                "mean": mean, "std": std,
                "mean_over_std": mean / std if std else np.nan,
                "acf1": acf1, "band_95": band,
                "acf1_outside_band": bool(abs(acf1) > band) if n else False,
            })
    return pd.DataFrame(rows)


def proxy_assumption_summary(diagnostics: pd.DataFrame) -> str:
    """A one-paragraph verdict on the diagnostics, for the notebook."""
    lines = []
    for name, group in diagnostics.groupby("series", sort=False):
        whole = group[group["period"] == "whole sample"].iloc[0]
        flagged = group[(group["period"] != "whole sample") & group["acf1_outside_band"]]
        mean_ok = abs(whole["mean_over_std"]) < 0.05
        lines.append(
            f"{name}: mean is {whole['mean']:+.6f} against a standard deviation of "
            f"{whole['std']:.4f} ({abs(whole['mean_over_std']):.1%} of it), so the mean is "
            f"{'negligible' if mean_ok else 'NOT negligible'}. "
            f"Lag-1 autocorrelation over the whole sample is {whole['acf1']:+.3f} "
            f"against a 95% band of +/-{whole['band_95']:.3f}, so it is "
            f"{'outside' if whole['acf1_outside_band'] else 'inside'} the band. "
            + (f"Sub-periods outside the band: {', '.join(flagged['period'])}."
               if not flagged.empty else "No sub-period is outside the band.")
        )
    return "\n".join(lines)


def fss_comparison(panel: pd.DataFrame, names: list[str] = PRICE_NAMES) -> pd.DataFrame:
    """Compare the FSS robustness proxy with the plain realised variance.

    Reports how many weeks the FSS correction makes non-positive (and so
    unusable), and the correlation between the two proxies on the log scale,
    which is what the models predict. A correlation close to 1 means the
    correction would not change the modelling.
    """
    rows = []
    for name in names:
        adj, fss = panel[f"{name}_rv_adj"], panel[f"{name}_rv_fss"]
        usable = panel[f"{name}_n_days"] >= MIN_DAYS_PER_WEEK
        both = adj.notna() & fss.notna()
        rows.append({
            "series": name,
            "weeks_usable_n_days": int(usable.sum()),
            "rv_adj_available": int(adj.notna().sum()),
            "rv_fss_available": int(fss.notna().sum()),
            "rv_fss_non_positive": int((usable & fss.isna() & adj.notna()).sum()),
            "corr_log_rv_adj_log_rv_fss": float(
                np.corrcoef(np.log(adj[both]), np.log(fss[both]))[0, 1]),
            "corr_levels": float(np.corrcoef(adj[both], fss[both])[0, 1]),
        })
    return pd.DataFrame(rows)


def raw_eia_missing_weeks(names: list[str] = EIA_NAMES) -> pd.DataFrame:
    """Weeks missing from each raw EIA CSV, after that series' first observation.

    Two things can be missing: a week present with an empty value, and a week
    absent from the file altogether. Both are found by laying the series on a
    complete Friday grid running from its own first to its own last observation,
    so nothing before a series starts is counted as missing.

    Nothing is filled: this only reports the dates, for stage 3 to act on.
    """
    rows = []
    for name in names:
        values = read_series(name).sort_index()
        present = values.dropna()
        if present.empty:
            rows.append({"metric": name, "first_date": None, "last_date": None,
                         "expected_weeks": 0, "n_missing": 0, "missing_dates": ""})
            continue
        grid = pd.date_range(present.index.min(), present.index.max(), freq=WEEK_FREQ)
        on_grid = values.reindex(grid)
        missing = on_grid.index[on_grid.isna()]
        # weeks EIA dated off the Friday grid would show up here as well
        off_grid = values.index.difference(grid)
        rows.append({
            "metric": name,
            "first_date": present.index.min().date(),
            "last_date": present.index.max().date(),
            "expected_weeks": len(grid),
            "n_missing": len(missing),
            "n_dates_off_friday_grid": len(off_grid),
            "missing_dates": ", ".join(d.strftime("%Y-%m-%d") for d in missing),
        })
    return pd.DataFrame(rows)


def panel_eia_missing_weeks(panel: pd.DataFrame,
                            names: list[str] = EIA_NAMES) -> pd.DataFrame:
    """Weeks with a missing EIA value in the panel, after the column starts.

    The panel only holds the sample period, and a series that starts later (only
    `cushing_stocks`) is NaN before its first value by construction. Those rows
    are not missing data, so the count starts at each column's first non-missing
    week.
    """
    rows = []
    for name in names:
        values = panel[name]
        present = values.dropna()
        if present.empty:
            rows.append({"metric": name, "first_week": None, "n_weeks_after_start": 0,
                         "n_missing_after_start": 0, "missing_weeks": ""})
            continue
        after_start = values.loc[present.index.min():]
        missing = after_start.index[after_start.isna()]
        rows.append({
            "metric": name,
            "first_week": present.index.min().date(),
            "n_weeks_after_start": len(after_start),
            "n_missing_after_start": len(missing),
            "missing_weeks": ", ".join(d.strftime("%Y-%m-%d") for d in missing),
        })
    return pd.DataFrame(rows)


def log_rv_gap_to_next(panel: pd.DataFrame, name: str = "wti",
                       week: str = "2020-04-24") -> dict:
    """How far one week's log_rv stands above the next-highest week.

    On the log scale a difference is a ratio: a gap of 1.2 log points means the
    variance of that week was exp(1.2) times the next-worst. This is the number
    that says whether the most extreme week is an outlier the models will chase.
    """
    column = f"{name}_log_rv"
    ranked = panel[column].dropna().sort_values(ascending=False)
    target = pd.Timestamp(week)
    top_week, top_value = ranked.index[0], ranked.iloc[0]
    runner_week, runner_value = ranked.index[1], ranked.iloc[1]
    return {
        "series": name,
        "highest_week": top_week.date(),
        "highest_log_rv": round(float(top_value), 3),
        "next_week": runner_week.date(),
        "next_log_rv": round(float(runner_value), 3),
        "gap_log_points": round(float(top_value - runner_value), 3),
        "times_larger": round(float(np.exp(top_value - runner_value)), 2),
        "asked_week_is_highest": bool(top_week == target),
    }


def most_volatile_weeks(panel: pd.DataFrame, name: str = "wti", n: int = 10) -> pd.DataFrame:
    """The `n` weeks with the highest log realised variance."""
    columns = [f"{name}_log_rv", f"{name}_rvol_ann", f"{name}_n_days", f"{name}_ret_w"]
    return panel.nlargest(n, f"{name}_log_rv")[columns]
