import pandas as pd
from load_forecasting_cali.model import train_load_forecaster


def test_time_series_cv(tmp_path):
    """Test the expanding-window cross-validation mode."""
    input_csv = tmp_path / "data.csv"
    # Create enough data for 5 CV splits
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "DATE": dates, "hour": 0, "CAISO": range(50),
        "day_of_week": 0, "month": 1, "is_weekend": 0, "is_holiday": 0,
        "is_peak_hour": 0, "load_lag_24": 1.0, "load_lag_168": 1.0, "load_rolling_mean_24": 1.0
    })
    df.to_csv(input_csv, index=False)

    metrics = train_load_forecaster(
        input_csv, output_plot=None,
        validation="time_series_cv", cv_splits=3
    )
    assert "mae_std" in metrics
    assert len(metrics["fold_mae"]) == 3
