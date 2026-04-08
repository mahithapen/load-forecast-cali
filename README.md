# CAISO Load Forecasting

This project builds a reproducible pipeline to forecast hourly electricity load for CAISO. It merges raw CAISO load spreadsheets, engineers calendar and lag features, optionally adds weather features, and trains an XGBoost regressor for short-term load forecasting.

## Dataset

The raw dataset is CAISO historical hourly load Excel files located in `caiso_load_data/`. These files are downloaded from CAISO’s historical load reports and represent a real-world dataset required for ORIE 5270.

## Installation

From the project root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

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

Optional weather features:

```bash
load-forecast weather --input-file caiso_model_ready.csv --output-file caiso_final_dataset.csv
```

Train the model:

```bash
load-forecast train --input-file caiso_model_ready.csv --plot-file forecast_check.png
```

## Running Tests

```bash
pytest
```

## Project Structure

- `src/load_forecasting_cali/`: package source code
- `setupdata/`: legacy scripts (now thin wrappers)
- `models/`: legacy training script (now a thin wrapper)
- `tests/`: unit tests
