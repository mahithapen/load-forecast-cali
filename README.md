# CAISO Load Forecasting

This project builds a reproducible pipeline to forecast hourly electricity load for CAISO. It merges raw CAISO load spreadsheets, engineers calendar and lag features, optionally adds weather features, and trains an XGBoost regressor for short-term load forecasting.

## Dataset

The raw dataset is CAISO historical hourly load Excel files in `caiso_load_data/`, from CAISO’s historical load reports (ORIE 5270). You can place workbooks there manually or use the downloader below.

## Installation

From the project root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,scraper]"
```

Use `.[dev]` only if you do not need the scraper. The `scraper` extra adds `requests` and `beautifulsoup4` for `setupdata/scraper.py`.

## Download raw data

With the venv activated (use this project’s Python so HTTPS/SSL matches your environment):

```bash
python setupdata/scraper.py
```

That fetches monthly `.xlsx` files from CAISO’s [historical EMS hourly load](https://www.caiso.com/library/historical-ems-hourly-load) page into `caiso_load_data/`.

## Usage

Run the full feature pipeline (merge + calendar + lags):

```bash
load-forecast pipeline
```

Run steps individually:

```bash
load-forecast merge --input-dir caiso_load_data --output-file caiso_load_complete.csv
load-forecast calendar --input-file caiso_load_complete.csv --output-file caiso_features.csv
load-forecast lags --input-file caiso_features.csv --output-file caiso_model_ready.csv
```

Optional weather features (requires **internet**; uses [Meteostat](https://github.com/meteostat/meteostat-python) hourly data for LA and SF). Long histories are fetched in **chunks** so requests stay under Meteostat’s default hourly range limit (~3 years per call). The default `train` feature set does **not** include weather columns; extend `DEFAULT_FEATURES` in `model.py` if you want the model to use them.

```bash
load-forecast weather --input-file caiso_model_ready.csv --output-file caiso_final_dataset.csv
```

Train the model:

```bash
load-forecast train --input-file caiso_model_ready.csv --plot-file forecast_check.png
```

Validation options (time-ordered only): `--validation holdout-ratio` (default, `--test-ratio 0.2`), `--validation holdout-months` (fixed test window, `--test-months 6`), or `--validation time-series-cv` (expanding-window `TimeSeriesSplit`, `--cv-splits 5`; reports mean and std of MAE/MAPE across folds).

## Running Tests

```bash
pytest
```

## Project Structure

- `src/`: package source (import name `load_forecasting_cali` via setuptools `package-dir`)
- `setupdata/`: legacy scripts (now thin wrappers)
- `models/`: legacy training script (now a thin wrapper)
- `tests/`: unit tests
