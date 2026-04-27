import pandas as pd
from load_forecasting_cali.features import add_calendar_features, add_lag_features


def test_add_calendar_features(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "features.csv"

    # Sunday Jan 1, 2023 (New Year's Day - Holiday and Weekend)
    df = pd.DataFrame({"DATE": ["2023-01-01"], "HR": [1], "CAISO": [100]})
    df.to_csv(input_csv, index=False)

    add_calendar_features(input_csv, output_csv)
    result = pd.read_csv(output_csv)

    assert result["is_weekend"].iloc[0] == 1
    assert result["is_holiday"].iloc[0] == 1
    assert "day_of_week" in result.columns


def test_add_lag_features(tmp_path):
    input_csv = tmp_path / "features.csv"
    output_csv = tmp_path / "lags.csv"

    # Create 200 hours of data to satisfy the 168-hour lag requirement
    dates = pd.date_range("2023-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "DATE": dates.date,
        "hour": dates.hour,
        "CAISO": range(200)
    })
    df.to_csv(input_csv, index=False)

    add_lag_features(input_csv, output_csv)
    result = pd.read_csv(output_csv)

    assert "load_lag_24" in result.columns
    assert "load_lag_168" in result.columns
    assert not result["load_lag_168"].isna().any()
