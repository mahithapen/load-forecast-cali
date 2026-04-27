import pandas as pd

from load_forecasting_cali.model import tune_load_forecaster


def test_tune_load_forecaster_returns_best(tmp_path):
    input_csv = tmp_path / "ready.csv"
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "DATE": dates,
            "hour": 0,
            "CAISO": [x % 10 for x in range(60)],
            "day_of_week": 0,
            "month": dates.month,
            "is_weekend": 0,
            "is_holiday": 0,
            "is_peak_hour": 0,
            "load_lag_24": 1.0,
            "load_lag_168": 1.0,
            "load_rolling_mean_24": 1.0,
        }
    )
    df.to_csv(input_csv, index=False)

    out = tune_load_forecaster(
        input_csv,
        time_series_cv_splits=3,
        n_estimators_grid=(5, 10),
        learning_rate_grid=(0.1,),
        max_depth_grid=(2, 3),
        random_state=0,
    )

    assert out["validation"] == "time_series_cv_grid_search"
    assert out["time_series_cv_splits"] == 3
    assert "best" in out
    assert "results" in out
    assert len(out["results"]) == 4
    assert "params" in out["best"]
