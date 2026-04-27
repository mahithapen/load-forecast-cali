# CAISO Load Forecasting

This project builds a reproducible pipeline to forecast hourly electricity load for the California Independent System Operator (CAISO). It merges raw CAISO load spreadsheets, engineers calendar and lag features, optionally adds weather data, and trains an XGBoost regressor for short-term forecasting.

## Dataset
The raw dataset consists of CAISO historical hourly load Excel files. By default, raw files live under `data/raw/` and derived datasets live under `data/processed/` (both are ignored by git).

## Installation
From the project root, set up your environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,scraper]"
```

* **`.[dev]`**: Installs development tools including `pytest` and `pytest-cov`.
* **`.[scraper]`**: Installs `requests` and `beautifulsoup4` required for the data downloader.

## Download Raw Data
With the virtual environment activated, run the scraper to fetch monthly `.xlsx` files from CAISO’s historical library:

```bash
python setupdata/scraper.py
```

## Usage
The pipeline is accessible via the `load-forecast` command, configured through `pyproject.toml`.

### Running the Full Pipeline
Execute the standard sequence (merge → calendar → lags) in one command:
```bash
load-forecast pipeline
```

### Individual Steps
For granular control, run steps individually:
* **Merge Data**: `load-forecast merge --input-dir data/raw/caiso_load_data --output-file data/processed/caiso_load_complete.csv`
* **Calendar Features**: `load-forecast calendar --input-file data/processed/caiso_load_complete.csv --output-file data/processed/caiso_features.csv`
* **Lag Features**: `load-forecast lags --input-file data/processed/caiso_features.csv --output-file data/processed/caiso_model_ready.csv`
* **Weather (Optional)**: `load-forecast weather --input-file data/processed/caiso_model_ready.csv --output-file data/processed/caiso_final_dataset.csv`
* **Train Model**: `load-forecast train --input-file data/processed/caiso_model_ready.csv --plot-file artifacts/forecast_check.png`

## Validation Modes
The training module supports multiple strategies to ensure forecast reliability:
* **`holdout-ratio`**: Chronological split using the last fraction of data for testing (default).
* **`holdout-months`**: Uses the last $N$ calendar months as the test set.
* **`time-series-cv`**: Employs an expanding-window `TimeSeriesSplit` and reports mean/std across folds.

## Running Tests
The project is configured for rigorous testing to maintain high code quality. To run the test suite and view the coverage report:

```bash
pytest
```

### Latest coverage result
Most recent local run (with `pytest-cov`) achieved **82% total coverage**:

```text
TOTAL               273     49    82%
8 passed in 17.97s
```

Coverage targets and testing paths are defined in `pyproject.toml`.

## Project Structure
* **`src/`**: Core package source code, utilizing relative imports for internal modularity.
* **`tests/`**: Unit tests, including mocks for the Meteostat API.
* **`setupdata/`**: Scripts for data scraping, cleaning, and initial feature setup.
* **`models/`**: Legacy training script wrappers.
* **`data/`**: Local datasets (ignored by git): `data/raw/`, `data/processed/`, `data/sample/`.
* **`artifacts/`**: Generated outputs (ignored by git): plots, model files, etc.