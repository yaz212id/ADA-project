"""Losses and tests for comparing volatility forecasts (used by stage 4).

The models are fitted on `log_rv = ln(rv_adj)`, because the log of realised
variance is close to normal and additive. But forecasts must NOT be compared on
that scale. Realised variance is a *noisy proxy* for the volatility we actually
care about, and Patton (2011) shows that with a noisy proxy only a few loss
functions rank forecasts in the same order as the (unobservable) true
volatility would. Those "robust" losses are QLIKE and MSE **on the variance
scale**. Mean squared error on log variance, and mean absolute error, are not
robust: they can rank a worse forecast above a better one.

So the workflow of stage 4 is:

1. predict `log_rv`;
2. convert the prediction back to a variance with `to_variance`, which adds the
   lognormal correction `exp(mu + s2/2)` — without it every forecast is biased
   downwards;
3. score it with `qlike` (the main loss), `mse`, and `mincer_zarnowitz` for
   bias, using the GLS version of Patton & Sheppard (2009).

All functions take the **variance scale**. Passing log values is a silent and
expensive mistake, so the inputs are checked and negative numbers are refused.

References
----------
Andersen, Bollerslev, Diebold & Labys (2003), "Modeling and forecasting
    realized volatility", Econometrica 71(2), 579-625.
Patton (2011), "Volatility forecast comparison using imperfect volatility
    proxies", Journal of Econometrics 160(1), 246-256.
Patton & Sheppard (2009), "Evaluating volatility and correlation forecasts",
    in: Handbook of Financial Time Series, 801-838.

Written with the help of Claude (Anthropic), which was asked to produce the
forecast evaluation code for this project (declared in the paper appendix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_variance_inputs(rv, h, names: tuple[str, str] = ("rv", "h")
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Align two variance series, drop missing pairs and refuse log-scale input.

    Realised variance and a variance forecast are always strictly positive, so
    a negative value means log values were passed by mistake — the most likely
    error in this pipeline, and one that would otherwise go unnoticed.
    """
    rv_name, h_name = names
    if isinstance(rv, pd.Series) and isinstance(h, pd.Series):
        joined = pd.concat([rv.rename("rv"), h.rename("h")], axis=1).dropna()
        rv_values, h_values = joined["rv"].to_numpy(), joined["h"].to_numpy()
    else:
        rv_values = np.asarray(rv, dtype="float64").ravel()
        h_values = np.asarray(h, dtype="float64").ravel()
        if rv_values.shape != h_values.shape:
            raise ValueError(
                f"{rv_name} and {h_name} have different lengths "
                f"({rv_values.size} and {h_values.size})"
            )
        keep = np.isfinite(rv_values) & np.isfinite(h_values)
        rv_values, h_values = rv_values[keep], h_values[keep]

    if rv_values.size == 0:
        raise ValueError("no overlapping non-missing observations to score")

    for values, name in ((rv_values, rv_name), (h_values, h_name)):
        if np.any(values < 0):
            raise ValueError(
                f"{name} contains negative values, so it looks like a LOG variance. "
                "These functions take the variance scale: convert a log forecast "
                "with to_variance() first."
            )
    if np.any(h_values <= 0):
        raise ValueError(
            f"{h_name} contains values <= 0; a variance forecast must be strictly "
            "positive (QLIKE divides by it and takes its log)."
        )
    if np.any(rv_values <= 0):
        raise ValueError(
            f"{rv_name} contains values <= 0; the realised variance proxy must be "
            "strictly positive (stage 2 already sets those weeks to NaN)."
        )
    return rv_values, h_values


def to_variance(log_pred, s2: float):
    """Turn a forecast of log variance into a forecast of variance.

    If `ln(rv)` is predicted as `mu` with normally distributed errors of
    variance `s2`, then the mean of `rv` is `exp(mu + s2 / 2)`, not `exp(mu)`.
    Taking `exp(mu)` alone predicts the *median* and therefore understates the
    variance every time. `s2` is the variance of the model's residuals on the
    log scale, measured on the training data.

    Pass `s2=0` only to get the median forecast on purpose.
    """
    if s2 < 0:
        raise ValueError("s2 is a variance and cannot be negative")
    if isinstance(log_pred, pd.Series):
        return np.exp(log_pred + s2 / 2)
    return np.exp(np.asarray(log_pred, dtype="float64") + s2 / 2)


def qlike(rv, h) -> float:
    """QLIKE loss: mean of `rv/h - ln(rv/h) - 1`. Lower is better. MAIN loss.

    This is Patton's (2011) QLIKE shifted so that a perfect forecast scores
    exactly 0; the shift is a constant, so it ranks forecasts identically to
    the usual `ln(h) + rv/h`. It is robust to noise in the proxy, and it
    penalises *under*-prediction far more than over-prediction, which is the
    behaviour wanted for a risk measure.
    """
    rv_values, h_values = _check_variance_inputs(rv, h)
    ratio = rv_values / h_values
    return float(np.mean(ratio - np.log(ratio) - 1))


def mse(rv, h) -> float:
    """Mean squared error on the variance scale: mean of `(rv - h)^2`.

    Also robust to proxy noise (Patton, 2011), but dominated by the few most
    volatile weeks, so it is reported next to QLIKE rather than instead of it.
    Note this is MSE of the *variance*, not of `log_rv`, which is not robust.
    """
    rv_values, h_values = _check_variance_inputs(rv, h)
    return float(np.mean((rv_values - h_values) ** 2))


def r2_variance(rv, h) -> float:
    """Share of the variance of `rv` explained, on the variance scale.

    `1 - MSE / var(rv)`, the course's R-squared applied to the variance scale so
    that it matches the losses above. It can be negative, which simply means the
    forecast does worse than predicting the sample mean.
    """
    rv_values, h_values = _check_variance_inputs(rv, h)
    denominator = np.mean((rv_values - rv_values.mean()) ** 2)
    return float(1 - np.mean((rv_values - h_values) ** 2) / denominator)


def mincer_zarnowitz(rv, h) -> pd.DataFrame:
    """Regress `rv = a + b*h + e` to test whether a forecast is unbiased.

    A forecast is unbiased when `a = 0` and `b = 1`. The function returns both:

    * **OLS** — the textbook version;
    * **GLS** — weighted least squares with weights `1/h^2`, as recommended by
      Patton & Sheppard (2009). The errors of a variance forecast grow with the
      level of the forecast, so OLS is dominated by the most volatile weeks;
      weighting by `1/h^2` makes the test informative about the whole sample.

    Columns: `a`, `b`, `r2`, `p_joint` (p-value of the joint test `a = 0` and
    `b = 1`) and `n`. A `p_joint` below 0.05 rejects unbiasedness. For the GLS
    row, `r2` is the weighted R-squared and is not comparable with the OLS one.
    """
    import statsmodels.api as sm  # imported here so stage 2 does not need it

    rv_values, h_values = _check_variance_inputs(rv, h)

    # A forecast that never changes is perfectly collinear with the intercept,
    # so a and b cannot be told apart and the test is not identified. This is a
    # real case — "predict the historical mean variance" is a natural benchmark —
    # so it is refused explicitly rather than returning a meaningless slope.
    spread = h_values.std() / abs(h_values.mean())
    if spread < 1e-8:
        raise ValueError(
            "the forecast is constant, so the Mincer-Zarnowitz regression is not "
            "identified (the intercept and the slope cannot be separated). "
            "Score a constant benchmark with qlike() and mse() instead."
        )

    # Variances are small numbers (about 0.003 for a weekly oil variance), which
    # makes X'X badly conditioned and the joint F-test numerically unreliable.
    # Scaling rv and h by the SAME constant leaves b and the joint test exactly
    # unchanged and only rescales a, which is undone below.
    scale = 1.0 / np.mean(rv_values)
    exog = pd.DataFrame({"const": 1.0, "forecast": h_values * scale})
    endog = rv_values * scale

    rows = {}
    for label, model in (
        ("OLS", sm.OLS(endog, exog)),
        ("GLS", sm.WLS(endog, exog, weights=1.0 / (h_values * scale) ** 2)),
    ):
        fit = model.fit()
        joint = fit.f_test("const = 0, forecast = 1")
        rows[label] = {
            "a": float(fit.params["const"]) / scale,   # back to the original units
            "b": float(fit.params["forecast"]),
            "r2": float(fit.rsquared),
            "p_joint": float(np.squeeze(joint.pvalue)),
            "n": int(fit.nobs),
        }
    return pd.DataFrame(rows).T[["a", "b", "r2", "p_joint", "n"]]


def compare_forecasts(rv, forecasts: dict[str, object]) -> pd.DataFrame:
    """Score several forecasts of the same target, best QLIKE first.

    `forecasts` maps a model name to its variance forecast. Returns one row per
    model with QLIKE, MSE, variance-scale R-squared, and the GLS Mincer-Zarnowitz
    slope and p-value.
    """
    rows = []
    for name, h in forecasts.items():
        try:
            table = mincer_zarnowitz(rv, h)
            slope, p_value = table.loc["GLS", "b"], table.loc["GLS", "p_joint"]
        except ValueError:
            # a constant benchmark has no identified regression; the losses still rank it
            slope, p_value = np.nan, np.nan
        rows.append({
            "model": name,
            "qlike": qlike(rv, h),
            "mse": mse(rv, h),
            "r2_variance": r2_variance(rv, h),
            "mz_gls_b": slope,
            "mz_gls_p": p_value,
        })
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Self-check: run with `python -m src.evaluation`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)

    # A realistic target: weekly variance built from 5 daily returns, so the
    # proxy is conditionally unbiased but noisy, exactly as in the project.
    n = 4000
    true_variance = np.exp(rng.normal(-5.0, 0.8, n))          # latent weekly variance
    rv = true_variance * rng.chisquare(5, n) / 5              # the proxy, E[rv | h] = h

    print("1. A perfect forecast scores 0 under QLIKE")
    print(f"   qlike(rv, rv) = {qlike(rv, rv):.12f}")
    assert abs(qlike(rv, rv)) < 1e-12

    print("\n2. QLIKE punishes under-prediction more than over-prediction")
    too_low = qlike(rv, rv / 2)      # forecast half the truth
    too_high = qlike(rv, rv * 2)     # forecast twice the truth
    print(f"   forecast = rv / 2 (too low) : {too_low:.6f}")
    print(f"   forecast = rv * 2 (too high): {too_high:.6f}")
    print(f"   under-prediction is {too_low / too_high:.2f} times worse")
    assert too_low > too_high
    # MSE on the variance scale does NOT share this asymmetry in the same way
    print(f"   for comparison, mse too low = {mse(rv, rv / 2):.3e}, "
          f"too high = {mse(rv, rv * 2):.3e}")

    print("\n3. Mincer-Zarnowitz recovers b = 1 for an unbiased forecast")
    table = mincer_zarnowitz(rv, true_variance)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    assert abs(table.loc["OLS", "b"] - 1) < 0.1, "OLS slope should be close to 1"
    assert abs(table.loc["GLS", "b"] - 1) < 0.1, "GLS slope should be close to 1"
    assert table.loc["GLS", "p_joint"] > 0.05, "an unbiased forecast must not be rejected"

    print("\n4. A biased forecast IS rejected (forecast = 0.7 * truth)")
    biased = mincer_zarnowitz(rv, 0.7 * true_variance)
    print(biased.to_string(float_format=lambda x: f"{x:.4f}"))
    assert biased.loc["GLS", "p_joint"] < 0.05

    print("\n5. The lognormal correction removes the downward bias")
    s2 = 0.8 ** 2
    log_forecast = np.log(true_variance) - s2 / 2      # a prediction of the log mean
    median_only = to_variance(log_forecast, s2=0.0)    # exp(mu): the median
    corrected = to_variance(log_forecast, s2=s2)       # exp(mu + s2/2): the mean
    print(f"   mean of rv                     : {rv.mean():.6f}")
    print(f"   mean forecast without correction: {median_only.mean():.6f}")
    print(f"   mean forecast with correction   : {corrected.mean():.6f}")
    print(f"   qlike without correction: {qlike(rv, median_only):.6f}")
    print(f"   qlike with correction   : {qlike(rv, corrected):.6f}")
    assert qlike(rv, corrected) < qlike(rv, median_only)

    print("\n6. Log-scale input is refused")
    try:
        qlike(rv, np.log(true_variance))
    except ValueError as exc:
        print(f"   ValueError: {str(exc)[:90]}...")
    else:
        raise AssertionError("passing log values should have raised")

    print("\n7. A constant forecast is refused by Mincer-Zarnowitz, not fudged")
    try:
        mincer_zarnowitz(rv, np.full(n, rv.mean()))
    except ValueError as exc:
        print(f"   ValueError: {str(exc)[:95]}...")
    else:
        raise AssertionError("a constant forecast should have raised")

    print("\n8. Ranking several forecasts")
    print(compare_forecasts(rv, {
        "perfect": true_variance,
        "too low (x0.5)": true_variance * 0.5,
        "too high (x2)": true_variance * 2,
        "constant (no MZ)": np.full(n, rv.mean()),
    }).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nAll self-checks passed.")
