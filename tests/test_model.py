import pytest
import pandas as pd
from load_forecasting_cali.model import train_load_forecaster


def test_train_load_forecaster(tmp_path):
    input_csv = tmp_path / "ready.csv"

    # Create dummy dataset with enough rows for a split
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "DATE": dates.date,
        "hour": dates.hour,
        "CAISO": [x % 10 for x in range(100)],
        "day_of_week": [1]*100,
        "month": [1]*100,
        "is_weekend": [0]*100,
        "is_holiday": [0]*100,
        "is_peak_hour": [0]*100,
        "load_lag_24": [1.0]*100,
        "load_lag_168": [1.0]*100,
        "load_rolling_mean_24": [1.0]*100
    })
    df.to_csv(input_csv, index=False)

    metrics = train_load_forecaster(
        input_csv, output_plot=None, validation="holdout_ratio", test_ratio=0.2)

    assert "mae" in metrics
    assert "mape" in metrics
    assert metrics["validation"] == "holdout_ratio"
