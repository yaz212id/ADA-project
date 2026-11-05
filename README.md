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
│   └── data_utils.py           list of EIA series; download, xls→csv conversion, reading
├── notebooks/                  one notebook per stage, run in order
│   └── 01_download_data.ipynb  stage 1: obtain the data
├── data/
│   ├── raw/
│   │   ├── xls/                original EIA downloads (not committed)
│   │   ├── weekly/             weekly petroleum data, one CSV per metric (committed)
│   │   ├── prices/             daily WTI and Brent prices as CSV (not committed)
│   │   └── manifest.csv        coverage of every file (written by notebook 01)
│   └── processed/              cleaned data (later stages)
├── results/
│   ├── figures/                every figure in the paper
│   └── tables/                 every table in the paper
└── paper/                      the paper (PDF)
```

Planned stages: `02_clean_data` → `03_features` (PCA, regimes) → `04_models` → `05_backtest` → `06_figures_tables`.

## How to run

1. In the provided cloud workspace, clone this repository.
2. Install the packages: `pip install -r requirements.txt`
3. Run the notebooks in order, each from top to bottom in a fresh kernel.

**Running time:** stage 1 takes about 1–2 minutes (13 small downloads). **Resources:** nothing beyond the provided workspace; stage 1 needs internet access to `www.eia.gov`.

## Data

All data come from the U.S. Energy Information Administration (EIA). No account or API key is needed. Notebook 01 downloads each series' history file from `https://www.eia.gov/dnav/pet/hist_xls/<code><d|w>.xls` and converts it to one CSV per metric (columns `date, <metric>`).

| CSV file | EIA code | Series | Units | Frequency | From | In repo |
|---|---|---|---|---|---|---|
| `prices/wti.csv` | RWTC | WTI spot price, Cushing OK | USD/barrel | daily | 1986 | no |
| `prices/brent.csv` | RBRTE | Brent spot price, Europe | USD/barrel | daily | 1987 | no |
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

### Licence

- **Weekly petroleum data** (Weekly Petroleum Status Report) are U.S. government data in the public domain ([EIA copyright and reuse policy](https://www.eia.gov/about/copyrights_reuse.php)). They are included in `data/raw/weekly/`. Source: U.S. Energy Information Administration, Weekly Petroleum Status Report.
- **Spot prices** are republished by EIA with permission from a third-party provider (LSEG/Refinitiv), so the raw price files are **not** included in this repository. Notebook 01 downloads them from EIA free of charge; `data/raw/manifest.csv` records the coverage of the files used. Source: U.S. Energy Information Administration, Spot Prices.

### If you downloaded the `.xls` files by hand

From the repository root, run (in the VS Code terminal, for example):

```
python -m src.data_utils "path/to/folder/with/the/xls/files"
```

Each file is recognised from the EIA code stored inside it, so the file names do not matter. It is saved as the matching CSV above; other files are skipped and listed. The original `.xls` files are not changed. Copying the `.xls` files into `data/raw/xls/` and running notebook 01 does the same.

### Notes for the later stages

- A weekly EIA value refers to the week ending **Friday** and is published the following **Wednesday** at 10:30 ET (Thursday after a US federal holiday). Features must only use values published before the week being forecast.
- Cushing stocks start in 2004, later than the other series.
- The WTI spot price was negative (−36.98 USD) on 20 April 2020.

## Tools and attribution

Generative AI was used to help write code and text; every tool is declared in the paper appendix. Code produced with a tool is marked at the top of the file concerned.
