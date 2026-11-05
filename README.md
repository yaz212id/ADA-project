# Oil-market fundamentals and crude oil volatility

ADA 2026 semester project · HEC Lausanne · *[Your full name]*

**Research question.** Can weekly data on the US oil market (how much oil is stored, produced and refined) help predict oil price volatility better than past prices alone, and can these forecasts lead to better trading decisions?

## Repository structure

```
ada-oil-volatility/
├── README.md
├── requirements.txt            packages needed to run the code
├── src/                        code shared by the notebooks
│   ├── config.py               folders, sample dates, train/tune/test split
│   ├── data_utils.py           list of EIA series; download, xls→csv conversion, reading
│   ├── cleaning.py             daily returns, weekly aggregation, EIA release-date alignment
│   └── evaluation.py           QLIKE, MSE and Mincer-Zarnowitz for scoring forecasts (stage 4)
├── notebooks/                  one notebook per stage, run in order
│   ├── 01_download_data.ipynb  stage 1: obtain the data
│   └── 02_clean_data.ipynb     stage 2: clean and build the weekly panel
├── data/
│   ├── raw/
│   │   ├── xls/                original EIA downloads (not committed)
│   │   ├── weekly/             weekly petroleum data, one CSV per metric (committed)
│   │   ├── prices/             daily WTI and Brent prices as CSV (not committed)
│   │   └── manifest.csv        coverage of every file (written by notebook 01)
│   └── processed/              cleaned data, not committed (written by notebook 02)
│       ├── daily_returns.csv   daily log returns of WTI and Brent
│       └── weekly_panel.csv    one row per week ending Friday: the stage 2 output
├── results/
│   ├── figures/                every figure in the paper
│   └── tables/                 every table in the paper
└── paper/                      the paper (PDF)
```

Planned stages: `03_features` (PCA, regimes) → `04_models` → `05_backtest` → `06_figures_tables`.

## How to run

1. In the provided cloud workspace, clone this repository.
2. Install the packages: `pip install -r requirements.txt`
3. **Get a free EIA API key** and put it in a file called `.env` (see *The EIA API key* below). The daily prices cannot be downloaded without it.
4. Run the notebooks in order, each from top to bottom in a fresh kernel.

| Stage | Notebook | What it produces | Running time |
|---|---|---|---|
| 1 | `01_download_data.ipynb` | the 13 CSV files in `data/raw/` and `manifest.csv` | 1–2 min (13 `.xls` downloads + ~12 API requests) |
| 2 | `02_clean_data.ipynb` | `data/processed/daily_returns.csv`, `data/processed/weekly_panel.csv`, `results/tables/data_coverage.csv` | under 15 s |

**Resources:** nothing beyond the provided workspace. Only stage 1 needs internet access (to `www.eia.gov` and `api.eia.gov`); stage 2 runs offline on the files stage 1 wrote. Peak memory is a few hundred MB — the whole data set is well under 10 MB.

### The EIA API key

The daily spot prices come from the **EIA API v2**, which needs a free key (no cost, no approval wait):

1. Register at <https://www.eia.gov/opendata/register.php>; the key arrives by email.
2. In the repository root, create a file called `.env` containing exactly one line:

   ```
   EIA_API_KEY=your_key_here
   ```

3. `.env` is listed in `.gitignore`, so the key is **never committed**. It is read by `python-dotenv` into the environment variable `EIA_API_KEY` and used only in requests to `api.eia.gov`. No notebook, CSV, table or log in this repository ever prints it.

An API key identifies the caller so EIA can apply its rate limits; it is not a password and gives access to no private data, but it is personal, so it is kept out of Git like any other credential.

## Data

All data come from the U.S. Energy Information Administration (EIA), from two of its products:

- the **11 weekly series** are read from the `.xls` history files at `https://www.eia.gov/dnav/pet/hist_xls/<code>w.xls` (no key needed);
- the **2 daily price series** come from the **EIA API v2** (`https://api.eia.gov/v2/petroleum/pri/spt`), which needs a free key.

The daily `.xls` price files are **not** used, because they silently omit hundreds of ordinary trading days (757 for WTI, 900 for Brent, worst in 1999, 2003 and 2011). The API returns the complete series, with identical prices on the dates both sources cover. Notebook 01 reports that comparison. Every series is saved as one CSV per metric (columns `date, <metric>`).

| CSV file | EIA code | Series | Units | Frequency | From | In repo |
|---|---|---|---|---|---|---|
| `prices/wti.csv` | RWTC | WTI spot price, Cushing OK (**EIA API v2**) | USD/barrel | daily | 1986 | no |
| `prices/brent.csv` | RBRTE | Brent spot price, Europe (**EIA API v2**) | USD/barrel | daily | 1987 | no |
| `weekly/crude_stocks.csv` | WCESTUS1 | Commercial crude oil stocks (excl. SPR) | thousand barrels | weekly | 1982 | yes |
| `weekly/cushing_stocks.csv` | W_EPC0_SAX_YCUOK_MBBL | Cushing OK crude oil stocks | thousand barrels | weekly | 2004 | yes |
| `weekly/spr_stocks.csv` | WCSSTUS1 | Strategic Petroleum Reserve stocks | thousand barrels | weekly | 1982 | yes |
| `weekly/gasoline_stocks.csv` | WGTSTUS1 | Total motor gasoline stocks | thousand barrels | weekly | 1990 | yes |
| `weekly/distillate_stocks.csv` | WDISTUS1 | Distillate fuel oil stocks | thousand barrels | weekly | 1982 | yes |
| `weekly/crude_production.csv` | WCRFPUS2 | Crude oil field production | thousand barrels/day | weekly | 1983 | yes |
| `weekly/refinery_input.csv` | WCRRIUS2 | Refiner net input of crude oil | thousand barrels/day | weekly | 1982 | yes |
| `weekly/refinery_utilization.csv` | WPULEUS3 | Refinery utilisation | % of operable capacity | weekly | 1990 | yes |
| `weekly/crude_imports.csv` | WCEIMUS2 | Commercial crude oil imports (excl. SPR) | thousand barrels/day | weekly | 1982 | yes |
| `weekly/crude_exports.csv` | WCREXUS2 | Crude oil exports | thousand barrels/day | weekly | 1991 | yes |
| `weekly/product_supplied.csv` | WRPUPUS2 | Product supplied, all products (demand proxy) | thousand barrels/day | weekly | 1990 | yes |

## References

- Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P. (2003). "Modeling and forecasting realized volatility." *Econometrica* 71(2), 579–625. — realised variance as the sum of squared log returns within a period, and modelling its logarithm.
- French, K. R., Schwert, G. W. & Stambaugh, R. F. (1987). "Expected stock returns and volatility." *Journal of Financial Economics* 19(1), 3–29. — building the same measure for a lower-frequency period out of daily returns, with the optional `2 * sum(r_t * r_{t-1})` correction for autocorrelation.
- Patton, A. J. (2011). "Volatility forecast comparison using imperfect volatility proxies." *Journal of Econometrics* 160(1), 246–256. — with a noisy proxy, only robust losses (QLIKE and MSE on the variance scale) rank forecasts correctly; MSE on log variance and MAE do not.
- Patton, A. J. & Sheppard, K. (2009). "Evaluating volatility and correlation forecasts." In: *Handbook of Financial Time Series*, 801–838. — a conditionally unbiased proxy suffices; use QLIKE and the GLS form of the Mincer–Zarnowitz regression.

**How this project uses them.** The target is weekly realised variance built from daily WTI log returns as in Andersen et al. (2003) and French, Schwert & Stambaugh (1987), modelled as `log_rv`; forecasts are converted back to variances with `exp(mu + s2/2)` and compared on the variance scale with QLIKE (main), MSE and the GLS Mincer–Zarnowitz regression, following Patton (2011) and Patton & Sheppard (2009).

### Licence

- **Weekly petroleum data** (Weekly Petroleum Status Report) are U.S. government data in the public domain ([EIA copyright and reuse policy](https://www.eia.gov/about/copyrights_reuse.php)). They are included in `data/raw/weekly/`. Source: U.S. Energy Information Administration, Weekly Petroleum Status Report.
- **Spot prices** are republished by EIA with permission from a third-party provider (LSEG/Refinitiv), so the raw price files are **not** included in this repository. Notebook 01 downloads them from EIA free of charge; `data/raw/manifest.csv` records the coverage of the files used. Source: U.S. Energy Information Administration, Spot Prices.

### If you downloaded the `.xls` files by hand

From the repository root, run (in the VS Code terminal, for example):

```
python -m src.data_utils "path/to/folder/with/the/xls/files"
```

Each file is recognised from the EIA code stored inside it, so the file names do not matter. It is saved as the matching CSV above; other files are skipped and listed. The original `.xls` files are not changed. Copying the `.xls` files into `data/raw/xls/` and running notebook 01 does the same.

### Stage 2: how the weekly panel is built

`weekly_panel.csv` has one row per week ending Friday (1 400 weeks, 2000-01-07 to 2026-10-30) and contains only information known at the close of that Friday.

- **Weeks** end on Friday (`W-FRI`), the same convention as the EIA weeks.
- **Price variables**, for WTI and for Brent: `rv` (realised variance, the sum of squared daily log returns), `n_days`, `rvol_ann` (`100*sqrt(52*rv)`), `log_rv`, `ret_w` (weekly log return). `rv`, `rvol_ann` and `log_rv` are `NaN` when `n_days < 3`.
- **Negative price.** WTI settled at −36.98 USD on 20 April 2020, so `ln(P)` is undefined: the returns into and out of that price are `NaN`. Prices are never filled or interpolated. Brent is the robustness check for that episode.
- **No look-ahead.** A weekly EIA value refers to the week ending Friday but is published the following **Wednesday** at 10:30 ET (Thursday after a US federal holiday). The panel uses `release_date = week-ending + 6 days` and a backward `merge_asof`, so week `t` only ever sees the EIA week ending `t − 7 days`. The `eia_week` column records which EIA week each row used.
- **Cushing stocks** start on 2004-04-09 and are `NaN` before that. No row or column is dropped.
- **The proxy.** Weekly realised variance is the sum of squared daily log returns (Andersen et al., 2003), built from daily returns as in French, Schwert & Stambaugh (1987), and rescaled to a five-day week as `rv_adj = rv * 5 / n_days` so holiday weeks stay comparable. Models are fitted on `log_rv = ln(rv_adj)`. With only ~5 returns a week the proxy is conditionally unbiased but noisy, which caps every model's R², so forecasts are scored on the **variance** scale with QLIKE, MSE and the GLS Mincer–Zarnowitz regression (`src/evaluation.py`), never with MSE on `log_rv` (Patton, 2011). `rv_fss` adds the FSS autocovariance correction and is kept as a robustness proxy only.
- **Remaining limitations.** No free intraday oil price data exists, so the weekly proxy cannot be made less noisy. Cushing stocks start in 2004. Section E of notebook 02 checks the proxy's assumptions, and `results/tables/data_coverage.csv` records the coverage of every column.

## Tools and attribution

Generative AI was used to help write code and text; every tool is declared in the paper appendix. Code produced with a tool is marked at the top of the file concerned.
