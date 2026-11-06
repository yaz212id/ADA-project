"""Stage 3: build the forecast target and the features.

One row per week ending Friday. Every value in a row is known at the close of
that Friday, and the target is next week's volatility:

* **target** `y = log_rv` of week t+1 (WTI; Brent as a robustness check), with
  `y_var = rv_adj` of week t+1 kept as well, because stage 4 scores forecasts on
  the variance scale;
* **HAR inputs** past volatility over 1, 4 and 12 weeks (Corsi, 2009, in logs);
* **12 oil-market features** from the EIA weekly series, each aligned so that
  week t only sees an EIA week published on or before Friday t;
* **an expiry flag** for the WTI futures contract, which is known years ahead.

Three rules hold everywhere in this module:

1. *No look-ahead.* The EIA alignment reuses `cleaning.align_eia_to_weeks`, so
   week t sees the EIA week ending t-7 or earlier. Seasonal averages use only
   *previous* years. The expiry calendar is published in advance.
2. *Training only.* The scaler, the clipping limits, the PCA and the Gaussian
   mixture are all fitted on training rows and then applied to every row.
3. *The feature list is frozen.* It lives in `config.OIL_FEATURES` and
   `config.BASE_FEATURES`; stage 4 adds nothing.

Written with the help of Claude (Anthropic), which was asked to produce the
feature construction, scaling, PCA and regime code for this project (declared
in the paper appendix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning as cl
from .config import (CHANGE_HORIZON_WEEKS, CLIP_PERCENTILES,
                     CONTINUOUS_FEATURES, DRIFT_FLAG_PCT,
                     EXPIRY_BUSINESS_DAYS_BEFORE,
                     EXPIRY_BUSINESS_DAYS_BEFORE_IF_HOLIDAY,
                     EXPIRY_REFERENCE_DAY, HAR_WINDOWS,
                     MAX_FORWARD_FILL_WEEKS, NEXT_WEEK_FIRST_DAY_OFFSET,
                     NEXT_WEEK_LAST_DAY_OFFSET, OIL_FEATURES,
                     PCA_VARIANCE_TARGET, RANDOM_STATE, REGIME_BIC_MARGIN,
                     REGIME_COVARIANCE_TYPE, REGIME_INPUTS, REGIME_K_VALUES,
                     REGIME_N_INIT, REGIME_STABILITY_MIN, REGIME_STABILITY_SEEDS,
                     SEASONAL_MIN_VALUES,
                     SEASONAL_WEEK_WINDOW, SEASONAL_YEARS, SPLIT_NAMES,
                     TRAIN_END, VALID_END, YEAR_HORIZON_WEEKS)
from .data_utils import read_series

# Every modelling constant lives in config.py; nothing is set here.
# The seeds used for the regime stability check are derived from how many are
# wanted, so the count is the only choice to make.
STABILITY_SEEDS = tuple(range(REGIME_STABILITY_SEEDS))


# ---------------------------------------------------------------------------
# 1. The seasonal average: "normal for this time of year", from the past only
# ---------------------------------------------------------------------------

def _week_positions(index: pd.DatetimeIndex) -> tuple[dict, list]:
    """Map (ISO year, ISO week) to its position on the weekly grid."""
    calendar = index.isocalendar()
    keys = list(zip(calendar["year"].astype(int), calendar["week"].astype(int)))
    return {key: position for position, key in enumerate(keys)}, keys


def _seasonal_positions(by_key: dict, iso_year: int, iso_week: int,
                        own_position: int, years: int, week_window: int) -> list[int]:
    """Positions of the comparable weeks in the previous `years` years.

    For each earlier year the same ISO week is located on the grid and its
    neighbours are taken **by position**, not by week number. That is what makes
    "the same week of the year, plus or minus one week" wrap correctly: for ISO
    week 1 the window reaches back into week 52 or 53 of the year before,
    instead of looking for a week 0 that does not exist. A year with only 52
    weeks is anchored on its week 52 when the target is week 53.

    Only positions strictly before the target week are returned, so a seasonal
    average can never contain the present or the future.
    """
    positions = []
    for back in range(1, years + 1):
        anchor = by_key.get((iso_year - back, iso_week))
        if anchor is None and iso_week >= 53:
            anchor = by_key.get((iso_year - back, 52))
        if anchor is None:
            continue
        for offset in range(-week_window, week_window + 1):
            candidate = anchor + offset
            if 0 <= candidate < own_position:
                positions.append(candidate)
    return positions


def seasonal_average(values: pd.Series,
                     years: int = SEASONAL_YEARS,
                     week_window: int = SEASONAL_WEEK_WINDOW,
                     min_values: int = SEASONAL_MIN_VALUES) -> pd.Series:
    """What this week of the year usually looks like, from earlier years only.

    For each week, averages the values observed in the same ISO week of the
    previous `years` years together with the `week_window` weeks either side, so
    up to 5 * 3 = 15 values. Weeks of the *current* year are never used, and the
    helper above only ever returns earlier positions, so the average cannot
    contain information from the week being described or from any later week.

    Fewer than `min_values` available values gives NaN rather than an average
    resting on too little history.
    """
    by_key, keys = _week_positions(values.index)
    data = values.to_numpy(dtype="float64")

    averages = []
    for own_position, (iso_year, iso_week) in enumerate(keys):
        positions = _seasonal_positions(by_key, iso_year, iso_week,
                                        own_position, years, week_window)
        collected = data[positions]
        collected = collected[~np.isnan(collected)]
        averages.append(collected.mean() if collected.size >= min_values else np.nan)

    return pd.Series(averages, index=values.index, name=f"{values.name}_seasonal")


def seasonal_average_sources(values: pd.Series, week: pd.Timestamp,
                             years: int = SEASONAL_YEARS,
                             week_window: int = SEASONAL_WEEK_WINDOW) -> pd.DataFrame:
    """The individual values behind one seasonal average, for the hand-check.

    Returns the date, ISO year and ISO week of every value used, so a reader can
    confirm that all of them come from earlier years.
    """
    by_key, keys = _week_positions(values.index)
    calendar = values.index.isocalendar()
    own_position = values.index.get_loc(week)
    iso_year = int(calendar.loc[week, "year"])
    iso_week = int(calendar.loc[week, "week"])

    positions = _seasonal_positions(by_key, iso_year, iso_week,
                                    own_position, years, week_window)
    return pd.DataFrame({
        "date": values.index[positions],
        "iso_year": [keys[i][0] for i in positions],
        "iso_week": [keys[i][1] for i in positions],
        "value": values.to_numpy()[positions],
    }).set_index("date")


# ---------------------------------------------------------------------------
# 2. The EIA series, on a complete weekly grid
# ---------------------------------------------------------------------------

def eia_history(names: list[str] | None = None,
                max_ffill: int = MAX_FORWARD_FILL_WEEKS) -> pd.DataFrame:
    """The full EIA weekly history on a complete Friday grid, lightly filled.

    The features need the whole history, not the 2000+ panel, because the
    seasonal averages reach five years back. Each series is laid on a complete
    grid of Fridays so that a week EIA skipped becomes a visible gap, then a gap
    of a single week is carried forward from the week before it. Longer gaps are
    left missing and the rows that depend on them are dropped later.

    Stage 2 reported where those gaps are: six weeks in 1982-83 for several
    series and one in 1991, all long before the modelling sample starts.
    """
    names = cl.EIA_NAMES if names is None else names
    series = {name: read_series(name) for name in names}
    first = min(s.dropna().index.min() for s in series.values())
    last = max(s.dropna().index.max() for s in series.values())
    grid = pd.date_range(first, last, freq=cl.WEEK_FREQ)

    filled = {}
    for name, values in series.items():
        on_grid = values.reindex(grid)
        # ffill uses the previous week only: never a future value
        filled[name] = on_grid.ffill(limit=max_ffill)
    return pd.DataFrame(filled).rename_axis("eia_week")


# ---------------------------------------------------------------------------
# 3. The 12 oil-market features, on the EIA week index
# ---------------------------------------------------------------------------

def build_oil_features(history: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 12 oil-market features, indexed by EIA week-ending Friday.

    Levels are deliberately avoided. US crude production more than doubled over
    the sample, exports went from nil to millions of barrels a day once the 1975
    export ban was lifted in late 2015, and the SPR was drawn down heavily in
    2022; a model fed levels would mostly learn those trends. Changes,
    surprises and deviations from the seasonal norm are comparable across the
    whole sample.

    Three shapes of feature:

    * **surprise** = this week's change minus the usual change for this week of
      the year — the news in the report;
    * **deviation** = level divided by the usual level for this week of the
      year, minus 1 — how scarce or plentiful oil is relative to normal, the
      quantity the theory of storage links to volatility (Pindyck, 2001; Geman
      & Ohana, 2009, find low crude inventories go with higher volatility);
    * **change** = a plain difference, for the series with too little history
      for a seasonal average (Cushing) or where a difference is the natural
      measure.
    """
    history = eia_history() if history is None else history
    features = {}

    # (a) surprises: the change in stocks against the usual change
    for name, column in (("crude", "crude_stocks"),
                         ("gasoline", "gasoline_stocks"),
                         ("distillate", "distillate_stocks")):
        change = history[column].diff().rename(f"{column}_change")
        features[f"{name}_surprise"] = change - seasonal_average(change)

        # (b) scarcity: the level against the usual level, as a fraction
        level = history[column]
        usual = seasonal_average(level)
        features[f"{name}_deviation"] = level / usual - 1

    # (c) flows and the other stocks
    features["cushing_change"] = history["cushing_stocks"].diff()
    features["spr_change"] = history["spr_stocks"].diff()
    # weekly production is a rounded estimate, so a 4-week difference is steadier
    features["production_change"] = history["crude_production"].diff(CHANGE_HORIZON_WEEKS)

    utilization = history["refinery_utilization"]
    features["utilization_surprise"] = utilization - seasonal_average(utilization)

    net_imports = (history["crude_imports"] - history["crude_exports"]).rename("net_imports")
    net_imports_average = net_imports.rolling(CHANGE_HORIZON_WEEKS).mean()
    features["net_imports_change"] = (net_imports_average
                                      - net_imports_average.shift(CHANGE_HORIZON_WEEKS))

    demand_average = history["product_supplied"].rolling(CHANGE_HORIZON_WEEKS).mean()
    features["demand_change"] = (demand_average
                                 / demand_average.shift(YEAR_HORIZON_WEEKS) - 1)

    frame = pd.DataFrame(features)[OIL_FEATURES]
    return frame.rename_axis("eia_week")


# ---------------------------------------------------------------------------
# 4. The WTI futures expiry calendar
# ---------------------------------------------------------------------------

def business_days(name: str = "wti") -> pd.DatetimeIndex:
    """The days the market traded, taken as the days with a WTI spot price."""
    return pd.DatetimeIndex(read_series(name).dropna().index)


def last_trading_days(days: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """The last trading day of every WTI front-month contract.

    The contract rule: trading stops **3 business days before the 25th calendar
    day of the month before delivery**, or **4 business days before the 25th**
    when the 25th is not a business day. So the May 2020 contract, whose
    delivery month is May, is governed by 25 April 2020 — a Saturday, hence four
    business days back, which is 21 April 2020: the day WTI settled at -36.98.

    Returns one row per delivery month with the reference 25th, whether it was a
    business day, and the last trading day.
    """
    days = business_days() if days is None else days
    trading = pd.DatetimeIndex(sorted(days))

    rows = []
    # a reference 25th in month M governs the contract delivered in month M+1
    months = pd.date_range(trading.min(), trading.max(), freq="MS")
    for month_start in months:
        reference = month_start + pd.Timedelta(days=EXPIRY_REFERENCE_DAY - 1)
        is_business = reference in trading
        before = trading[trading < reference]
        steps = (EXPIRY_BUSINESS_DAYS_BEFORE if is_business
                 else EXPIRY_BUSINESS_DAYS_BEFORE_IF_HOLIDAY)
        if len(before) < steps:
            continue
        last_day = before[-steps]
        delivery = (month_start + pd.offsets.MonthBegin(1))
        rows.append({
            "delivery_month": delivery.strftime("%Y-%m"),
            "reference_25th": reference,
            "25th_is_business_day": is_business,
            "business_days_back": steps,
            "last_trading_day": last_day,
        })
    return pd.DataFrame(rows)


def expiry_next_week(week_index: pd.DatetimeIndex,
                     expiries: pd.DataFrame | None = None) -> pd.Series:
    """1 when a contract's last trading day falls in week t+1, else 0.

    Week t+1 runs from the Monday after Friday t to the following Friday, that
    is from t+3 to t+7. The expiry calendar is published years ahead, so knowing
    at Friday t that a contract expires next week is not look-ahead.
    """
    expiries = last_trading_days() if expiries is None else expiries
    last_days = pd.DatetimeIndex(expiries["last_trading_day"])

    flags = []
    for week in week_index:
        start = week + pd.Timedelta(days=NEXT_WEEK_FIRST_DAY_OFFSET)
        end = week + pd.Timedelta(days=NEXT_WEEK_LAST_DAY_OFFSET)
        flags.append(int(((last_days >= start) & (last_days <= end)).any()))
    return pd.Series(flags, index=week_index, name="expiry_next_week", dtype="int64")


# ---------------------------------------------------------------------------
# 5. Targets and HAR inputs, from the stage 2 panel
# ---------------------------------------------------------------------------

def build_targets(panel: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Next week's volatility, on the log scale and on the variance scale.

    `y` is `log_rv` of week t+1 and `y_var` is `rv_adj` of week t+1, both shifted
    back by one row so that they sit next to the features that must predict
    them. Models are fitted on `y`; stage 4 scores them on `y_var`, because
    QLIKE needs the variance scale.
    """
    names = cl.PRICE_NAMES if names is None else names
    columns = {}
    for name in names:
        prefix = "" if name == "wti" else f"{name}_"
        columns[f"{prefix}y"] = panel[f"{name}_log_rv"].shift(-1)
        columns[f"{prefix}y_var"] = panel[f"{name}_rv_adj"].shift(-1)
    return pd.DataFrame(columns, index=panel.index)


def build_har_inputs(panel: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Past volatility over 1, 4 and 12 weeks, in logs (Corsi, 2009).

    Each column is the log of the average `rv_adj` over a window **ending at
    week t**, so it uses only volatility already observed. `har_w` is simply
    `log(rv_adj_t)`. Taking the log of the average, rather than the average of
    the logs, keeps the three inputs on the same scale as the target.
    """
    names = cl.PRICE_NAMES if names is None else names
    columns = {}
    for name in names:
        prefix = "" if name == "wti" else f"{name}_"
        variance = panel[f"{name}_rv_adj"]
        for column, window in HAR_WINDOWS.items():
            rolling = variance.rolling(window, min_periods=window).mean()
            columns[f"{prefix}{column}"] = np.log(rolling)
    return pd.DataFrame(columns, index=panel.index)


# ---------------------------------------------------------------------------
# 6. Splits
# ---------------------------------------------------------------------------

def assign_splits(week_index: pd.DatetimeIndex,
                  train_end: str = TRAIN_END,
                  valid_end: str = VALID_END) -> pd.Series:
    """Label each week train / validation / test, by the date of week t.

    The split is chronological and never random: a model is only ever trained on
    weeks that came before the weeks it is tested on.
    """
    labels = pd.Series("test", index=week_index, name="split", dtype="object")
    labels[week_index <= pd.Timestamp(train_end)] = "train"
    labels[(week_index > pd.Timestamp(train_end))
           & (week_index <= pd.Timestamp(valid_end))] = "validation"
    return labels


# ---------------------------------------------------------------------------
# 7. Scaling, drift check and clipping - all fitted on training rows only
# ---------------------------------------------------------------------------

def fit_standard_scaler(frame: pd.DataFrame, train_mask: pd.Series,
                        columns: list[str] | None = None):
    """Lecture 6: fit z = (x - mu) / sigma on the TRAINING rows only.

    Uses `sklearn.preprocessing.StandardScaler`, which divides by a standard
    deviation computed with 1/n — the same definition as the lecture, and not
    pandas' default 1/(n-1). Fitting on training rows alone is what keeps the
    validation and test years out of the model: the scaler stores the training
    mean and standard deviation and those same numbers are then applied to
    every row.

    Returns the fitted scaler and a small table of its parameters, so they can
    be saved to results/tables/scaler_params.csv and inspected.
    """
    from sklearn.preprocessing import StandardScaler

    columns = CONTINUOUS_FEATURES if columns is None else columns
    scaler = StandardScaler().fit(frame.loc[train_mask, columns])
    params = pd.DataFrame({
        "feature": columns,
        "train_mean": scaler.mean_,
        "train_std": scaler.scale_,
        "n_train": int(train_mask.sum()),
    })
    return scaler, params


def fit_scaler(frame: pd.DataFrame, train_mask: pd.Series,
               columns: list[str] | None = None) -> pd.DataFrame:
    """The training mean and standard deviation of each feature, as a table.

    A thin wrapper around `fit_standard_scaler` for the places that only need
    the numbers, such as the saved table and the notebook's check that the
    scaler really was fitted on the training rows.
    """
    return fit_standard_scaler(frame, train_mask, columns=columns)[1]


def apply_scaler(frame: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Standardise every feature with the stored training mean and deviation.

    Takes the parameter table rather than the fitted object, so a scaling can
    be reproduced from the saved CSV alone. `StandardScaler` already replaces a
    zero standard deviation with 1, and so does this.
    """
    scaled = {}
    for _, row in params.iterrows():
        name, std = row["feature"], row["train_std"]
        scaled[f"{name}_z"] = (frame[name] - row["train_mean"]) / (std if std else 1.0)
    return pd.DataFrame(scaled, index=frame.index)


def fit_clip_limits(scaled: pd.DataFrame, train_mask: pd.Series,
                    percentiles: tuple[float, float] = CLIP_PERCENTILES) -> pd.DataFrame:
    """The TRAINING percentile range used both for the drift check and clipping."""
    low, high = percentiles
    train = scaled.loc[train_mask]
    return pd.DataFrame({
        "feature": list(scaled.columns),
        "lower": train.quantile(low / 100).to_numpy(),
        "upper": train.quantile(high / 100).to_numpy(),
        "percentile_low": low,
        "percentile_high": high,
    })


def drift_report(scaled: pd.DataFrame, splits: pd.Series, limits: pd.DataFrame,
                 threshold: float = DRIFT_FLAG_PCT) -> pd.DataFrame:
    """Share of validation and test values outside the training range.

    A feature whose later values often sit outside the range the model was
    trained on has *drifted*: the model would be extrapolating. This is the
    honest way to find that out before stage 4, rather than after.
    `flagged` marks anything above `threshold` percent.
    """
    rows = []
    for _, limit in limits.iterrows():
        name = limit["feature"]
        values = scaled[name]
        outside = (values < limit["lower"]) | (values > limit["upper"])
        row = {"feature": name}
        for split in SPLIT_NAMES:
            mask = splits == split
            row[f"pct_outside_{split}"] = round(
                100 * float(outside[mask].mean()) if mask.any() else np.nan, 2)
        row["flagged"] = bool(
            max(row["pct_outside_validation"], row["pct_outside_test"]) > threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_clipping(scaled: pd.DataFrame, limits: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip each standardised feature to its training range.

    Without this, a linear model meeting a value far outside anything it was
    trained on produces an absurd forecast. Clipping caps the input instead, so
    an extreme week is treated as "extreme" rather than as a number the model
    has never seen. The targets are never clipped.

    Returns the clipped frame and a count of how many values were clipped in
    each split.
    """
    clipped = scaled.copy()
    counts = []
    for _, limit in limits.iterrows():
        name = limit["feature"]
        below = scaled[name] < limit["lower"]
        above = scaled[name] > limit["upper"]
        clipped[name] = scaled[name].clip(limit["lower"], limit["upper"])
        counts.append({"feature": name, "n_clipped_low": int(below.sum()),
                       "n_clipped_high": int(above.sum()),
                       "n_clipped": int((below | above).sum())})
    return clipped, pd.DataFrame(counts)


def clipped_counts_by_split(scaled: pd.DataFrame, limits: pd.DataFrame,
                            splits: pd.Series) -> pd.DataFrame:
    """How many values each feature had clipped, split by split."""
    rows = []
    for _, limit in limits.iterrows():
        name = limit["feature"]
        outside = (scaled[name] < limit["lower"]) | (scaled[name] > limit["upper"])
        row = {"feature": name}
        for split in SPLIT_NAMES:
            row[f"n_clipped_{split}"] = int(outside[splits == split].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. PCA on the oil-market features
# ---------------------------------------------------------------------------

def fit_pca(clipped: pd.DataFrame, train_mask: pd.Series,
            target: float = PCA_VARIANCE_TARGET):
    """Lecture 11: fit PCA on the TRAINING rows of the 12 oil-market features.

    Uses `sklearn.decomposition.PCA`. The loadings are `components_`, the
    eigenvectors of the training covariance matrix, and the variance shares are
    `explained_variance_ratio_`; no eigen-decomposition is written by hand.

    The 12 features overlap a great deal — three stock surprises move together,
    and so do the three deviations — so a handful of components carries most of
    the information. Only the oil features go in: the HAR inputs and the expiry
    flag are kept as they are, so the comparison in stage 4 still isolates the
    EIA data.

    Returns the fitted PCA, the number of components reaching `target`, and the
    number reaching 90%.
    """
    from sklearn.decomposition import PCA

    columns = [f"{name}_z" for name in OIL_FEATURES]
    model = PCA(random_state=RANDOM_STATE).fit(clipped.loc[train_mask, columns])
    cumulative = np.cumsum(model.explained_variance_ratio_)
    n_target = int(np.searchsorted(cumulative, target) + 1)
    n_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    return model, n_target, n_90, columns


def pca_scores(model, clipped: pd.DataFrame, columns: list[str],
               n_components: int) -> pd.DataFrame:
    """The component scores pc1..pck for every week."""
    scores = model.transform(clipped[columns])[:, :n_components]
    return pd.DataFrame(
        scores, index=clipped.index,
        columns=[f"pc{i}" for i in range(1, n_components + 1)])


def pca_loadings(model, columns: list[str], n_components: int) -> pd.DataFrame:
    """The loadings, i.e. how each feature contributes to each component."""
    loadings = pd.DataFrame(
        model.components_[:n_components].T,
        index=[c.removesuffix("_z") for c in columns],
        columns=[f"pc{i}" for i in range(1, n_components + 1)])
    return loadings.rename_axis("feature")


def pca_variance_table(model) -> pd.DataFrame:
    """Variance explained by each component, and cumulatively."""
    ratio = model.explained_variance_ratio_
    return pd.DataFrame({
        "component": [f"pc{i}" for i in range(1, len(ratio) + 1)],
        "explained_variance_pct": (100 * ratio).round(2),
        "cumulative_pct": (100 * np.cumsum(ratio)).round(2),
    })


def name_components(loadings: pd.DataFrame, n_top: int = 3) -> pd.DataFrame:
    """Suggest an economic reading of each component from its largest loadings.

    Only a suggestion: the paper should state the name and the loadings that
    justify it, so a reader can disagree.
    """
    rows = []
    for component in loadings.columns:
        column = loadings[component]
        top = column.abs().nlargest(n_top).index
        description = ", ".join(
            f"{name} {column[name]:+.2f}" for name in top)
        rows.append({"component": component, "top_loadings": description})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Regimes: a Gaussian mixture on past volatility and the weekly return
# ---------------------------------------------------------------------------

def regime_inputs(frame: pd.DataFrame, train_mask: pd.Series,
                  columns: list[str] | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise the mixture's inputs with TRAINING statistics.

    The inputs are week t's own past volatility (`har_w`) and weekly return
    (`wti_ret_w`), so a regime label uses nothing beyond Friday t.
    """
    columns = REGIME_INPUTS if columns is None else columns
    scaler, params = fit_standard_scaler(frame, train_mask, columns=columns)
    scaled = pd.DataFrame(scaler.transform(frame[columns]), index=frame.index,
                          columns=[f"{name}_z" for name in columns])
    return scaled, params


def regime_bic_table(scaled: pd.DataFrame, train_mask: pd.Series,
                     k_values: tuple[int, ...] = REGIME_K_VALUES,
                     margin: float = REGIME_BIC_MARGIN) -> tuple[pd.DataFrame, int]:
    """Lecture 12: fit the EM mixture for each K on TRAINING rows, choose K by BIC.

    Uses `sklearn.mixture.GaussianMixture`, which is fitted by the
    expectation-maximisation algorithm of the lecture, and its `.bic()`.
    scikit-learn reports BIC as -2 x the lecture's criterion, so **lower is
    better** here where the lecture maximises.

    BIC trades fit against the number of parameters. A difference of more than
    10 is "very strong" evidence for the better model (Kass & Raftery, 1995), so
    the rule is to take the **smallest** K whose BIC is within 10 points of the
    best: a more complicated model has to earn its extra regimes.
    """
    from sklearn.mixture import GaussianMixture

    train = scaled.loc[train_mask].to_numpy()
    rows = []
    for k in k_values:
        model = GaussianMixture(n_components=k,
                                covariance_type=REGIME_COVARIANCE_TYPE,
                                n_init=REGIME_N_INIT,
                                random_state=RANDOM_STATE).fit(train)
        rows.append({"k": k, "bic": float(model.bic(train)),
                     "aic": float(model.aic(train)),
                     "converged": bool(model.converged_)})
    table = pd.DataFrame(rows)
    best = table["bic"].min()
    table["bic_minus_best"] = (table["bic"] - best).round(2)
    chosen = int(table.loc[table["bic_minus_best"] <= margin, "k"].min())
    return table, chosen


def fit_regimes(scaled: pd.DataFrame, train_mask: pd.Series, n_regimes: int,
                random_state: int = RANDOM_STATE):
    """Lecture 12: fit the chosen EM mixture on TRAINING rows, ordered by calmness.

    The mixture labels components in an arbitrary order, which would make
    "regime 1" mean different things from one run to the next. They are
    therefore re-ordered by their mean standardised `har_w`, so regime 0 is
    always the calmest. Returns the model and the ordering.
    """
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(n_components=n_regimes,
                            covariance_type=REGIME_COVARIANCE_TYPE,
                            n_init=REGIME_N_INIT, random_state=random_state)
    model.fit(scaled.loc[train_mask].to_numpy())
    # column 0 of the inputs is har_w
    order = np.argsort(model.means_[:, 0])
    return model, order


def regime_labels(model, order: np.ndarray, scaled: pd.DataFrame) -> pd.DataFrame:
    """Regime probabilities and the most likely regime, for every week.

    Lecture 12: `predict_proba` gives the responsibilities of the EM algorithm,
    that is the probability that each week belongs to each regime.
    """
    probabilities = model.predict_proba(scaled.to_numpy())[:, order]
    columns = {f"regime_p{i}": probabilities[:, i] for i in range(probabilities.shape[1])}
    frame = pd.DataFrame(columns, index=scaled.index)
    frame["regime"] = probabilities.argmax(axis=1)
    return frame


def regime_stability(scaled: pd.DataFrame, train_mask: pd.Series, n_regimes: int,
                     seeds: tuple[int, ...] = STABILITY_SEEDS) -> dict:
    """Share of training weeks that get the same regime under every seed.

    A mixture is fitted by a search that can land in different places. If the
    labels move when the seed changes, the regimes are not a real feature of the
    data and should not be used. Each fit is re-ordered by calmness first, so the
    comparison is fair.
    """
    label_sets = []
    for seed in seeds:
        model, order = fit_regimes(scaled, train_mask, n_regimes, random_state=seed)
        labels = regime_labels(model, order, scaled.loc[train_mask])["regime"]
        label_sets.append(labels)

    stacked = pd.concat(label_sets, axis=1)
    agree = stacked.nunique(axis=1) == 1
    return {"n_seeds": len(seeds), "n_train_weeks": int(len(stacked)),
            "share_same_label": round(float(agree.mean()), 4),
            "n_disagreeing_weeks": int((~agree).sum())}


def regime_stability_table(scaled: pd.DataFrame, train_mask: pd.Series,
                           k_values: tuple[int, ...] = REGIME_K_VALUES,
                           seeds: tuple[int, ...] = STABILITY_SEEDS) -> pd.DataFrame:
    """Stability of every candidate K, not only the chosen one.

    `share_same_label` is the share of training weeks given the same regime
    number by all seeds. `mean_ari` is the adjusted Rand index against the first
    seed, which ignores how the regimes are numbered and so separates a genuine
    disagreement about which weeks belong together from a mere relabelling.
    Both near 1 mean the regimes are a real feature of the data; `stable` marks a
    K whose agreement reaches `REGIME_STABILITY_MIN`.
    """
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for k in k_values:
        label_sets = []
        for seed in seeds:
            model, order = fit_regimes(scaled, train_mask, k, random_state=seed)
            label_sets.append(
                regime_labels(model, order, scaled.loc[train_mask])["regime"].to_numpy())
        stacked = np.vstack(label_sets)
        same = float((stacked == stacked[0]).all(axis=0).mean())
        ari = float(np.mean([adjusted_rand_score(label_sets[0], label_sets[i])
                             for i in range(1, len(seeds))])) if len(seeds) > 1 else 1.0
        rows.append({"k": k, "share_same_label": round(same, 3),
                     "mean_ari": round(ari, 3),
                     "stable": bool(same >= REGIME_STABILITY_MIN)})
    return pd.DataFrame(rows)


def regime_summary(frame: pd.DataFrame, splits: pd.Series,
                   train_mask: pd.Series) -> pd.DataFrame:
    """Weeks per regime in each split, and next-week volatility on TRAINING rows.

    The average target is shown for training rows only: looking at how the
    target behaves by regime in the validation or test years would be exactly
    the kind of peeking stage 3 must not do.
    """
    rows = []
    for regime in sorted(frame["regime"].unique()):
        mask = frame["regime"] == regime
        row = {"regime": int(regime)}
        for split in SPLIT_NAMES:
            row[f"n_{split}"] = int((mask & (splits == split)).sum())
        train_rows = mask & train_mask
        row["mean_har_w_train"] = round(float(frame.loc[train_rows, "har_w"].mean()), 3)
        row["mean_next_week_y_train"] = round(float(frame.loc[train_rows, "y"].mean()), 3)
        row["mean_next_week_rvol_ann_train"] = round(
            float(100 * np.sqrt(52 * frame.loc[train_rows, "y_var"]).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. The main entry point
# ---------------------------------------------------------------------------

def read_panel(path=None) -> pd.DataFrame:
    """The stage 2 weekly panel, read from data/processed/weekly_panel.csv."""
    from .config import WEEKLY_PANEL_PATH

    path = WEEKLY_PANEL_PATH if path is None else path
    if not pd.io.common.file_exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run notebook 02 (or `python run_all.py --only 02`) first."
        )
    return pd.read_csv(path, parse_dates=["date", "eia_week"], index_col="date")


def build_features(extra_lag_weeks: int = 0, panel: pd.DataFrame | None = None,
                   n_regimes: int | None = None) -> dict:
    """Build the whole stage 3 table, and everything fitted along the way.

    `n_regimes` overrides the number of regimes. Left as None, the BIC rule of
    `regime_bic_table` decides. The override exists because the BIC rule and the
    stability check can disagree: BIC keeps improving with more regimes, while a
    K whose labels move when the random seed changes is not describing anything
    real. `regime_stability_table` reports both so the choice can be made on
    the evidence, and the result records which rule was used.

    `extra_lag_weeks` delays the EIA data by whole weeks. The default 0 is the
    real timing: week t sees the EIA week ending t-7, published on the Wednesday
    inside week t. Stage 4 calls this with `extra_lag_weeks=1` as a robustness
    check, so week t sees only the EIA week ending t-14 — if the results survive
    that, they do not depend on the exact release day.

    Returns a dict holding the features table and every object fitted on the
    training rows (scaler, clipping limits, PCA, mixture) plus the diagnostic
    tables the notebook prints.
    """
    panel = read_panel() if panel is None else panel
    lag_days = 6 + 7 * extra_lag_weeks

    # --- the parts that come from the stage 2 panel
    targets = build_targets(panel)
    har = build_har_inputs(panel)

    # --- the oil features, built on the full EIA history and then aligned
    oil_on_eia_weeks = build_oil_features()
    aligned = cl.align_eia_to_weeks(panel.index, oil_on_eia_weeks, lag_days=lag_days)

    # --- the expiry flag, known in advance
    flag = expiry_next_week(panel.index)

    frame = pd.concat(
        [targets, har, aligned, flag,
         panel[["wti_ret_w", "brent_ret_w", "wti_negative_price_week"]]],
        axis=1)

    # --- the modelling sample: every week where all of it exists
    # The modelling sample is defined by the MAIN series, WTI. Brent is a
    # robustness check: its one unusable week (2018-12-28, too few trading days)
    # would otherwise blank brent_har_q for the next 12 weeks and delete the
    # start of the WTI test period with it. Brent columns stay in the file with
    # NaN where they are unavailable, and the Brent run drops its own rows.
    required = (["y", "y_var", "har_w", "har_m", "har_q"]
                + OIL_FEATURES + ["expiry_next_week", "wti_ret_w"])
    complete = frame[required].notna().all(axis=1)
    if not complete.any():
        raise RuntimeError("no week has a complete set of features")
    sample_start = frame.index[complete][0]

    in_window = frame.index >= sample_start
    has_target = frame["y"].notna()
    candidate = in_window & has_target
    dropped = frame.index[candidate & ~complete]

    frame = frame.loc[candidate & complete].copy()
    splits = assign_splits(frame.index)
    frame["split"] = splits
    train_mask = splits == "train"

    # --- scaling, drift, clipping: fitted on training rows only
    # Lecture 6: standardise with StandardScaler, fitted on training rows only
    standard_scaler, scaler = fit_standard_scaler(frame, train_mask)
    scaled = pd.DataFrame(
        standard_scaler.transform(frame[CONTINUOUS_FEATURES]), index=frame.index,
        columns=[f"{name}_z" for name in CONTINUOUS_FEATURES])
    limits = fit_clip_limits(scaled, train_mask)
    drift = drift_report(scaled, splits, limits)
    clipped, clip_counts = apply_clipping(scaled, limits)
    clip_by_split = clipped_counts_by_split(scaled, limits, splits)

    # --- PCA on the oil features only
    pca_model, n_components, n_90, pca_columns = fit_pca(clipped, train_mask)
    scores = pca_scores(pca_model, clipped, pca_columns, n_components)
    loadings = pca_loadings(pca_model, pca_columns, n_components)

    # --- regimes
    regime_scaled, regime_scaler = regime_inputs(frame, train_mask)
    bic_table, k_by_bic = regime_bic_table(regime_scaled, train_mask)
    stability_table = regime_stability_table(regime_scaled, train_mask)
    chosen_k = k_by_bic if n_regimes is None else int(n_regimes)
    mixture, order = fit_regimes(regime_scaled, train_mask, chosen_k)
    regimes = regime_labels(mixture, order, regime_scaled)

    features = pd.concat([frame, clipped, scores, regimes], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    summary = regime_summary(features, splits, train_mask)
    stability = regime_stability(regime_scaled, train_mask, chosen_k)

    return {
        "features": features,
        "sample_start": sample_start,
        "dropped_rows": dropped,
        "splits": splits,
        "scaler": scaler,
        "clip_limits": limits,
        "drift": drift,
        "clip_counts": clip_counts,
        "clip_by_split": clip_by_split,
        "pca_model": pca_model,
        "pca_variance": pca_variance_table(pca_model),
        "pca_loadings": loadings,
        "pca_names": name_components(loadings),
        "n_components": n_components,
        "n_components_90": n_90,
        "regime_scaler": regime_scaler,
        "regime_bic": bic_table,
        "regime_stability_table": stability_table,
        "n_regimes": chosen_k,
        "n_regimes_by_bic": k_by_bic,
        "n_regimes_rule": "BIC" if n_regimes is None else "set by hand",
        "regime_summary": summary,
        "regime_stability": stability,
        "extra_lag_weeks": extra_lag_weeks,
        "lag_days": lag_days,
    }
