from __future__ import annotations

from pathlib import Path
import pandas as pd

from load_forecasting_cali.data import merge_caiso_data
from load_forecasting_cali.features import add_calendar_features, add_lag_features
from load_forecasting_cali.model import train_load_forecaster


def _make_hourly_df(start: str, periods: int) -> pd.DataFrame:
    dt = pd.date_range(start=start, periods=periods, freq="h")
    return pd.DataFrame(
        {
            "DATE": dt.date,
            "HR": dt.hour,
            "CAISO": range(periods),
        }
    )


def test_merge_caiso_data(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    df1 = _make_hourly_df("2024-01-01 00:00:00", 2)
    df2 = _make_hourly_df("2024-01-01 02:00:00", 2)
    df1.to_excel(input_dir / "a.xlsx", index=False)
    df2.to_excel(input_dir / "b.xlsx", index=False)

    output_file = tmp_path / "merged.csv"
    merged = merge_caiso_data(input_dir, output_file)

    assert output_file.exists()
    assert len(merged) == 4
    assert list(merged["HR"]) == [0, 1, 2, 3]


def test_add_calendar_features(tmp_path: Path) -> None:
    input_file = tmp_path / "merged.csv"
    df = _make_hourly_df("2024-01-06 00:00:00", 3)  # Saturday
    df.to_csv(input_file, index=False)

    output_file = tmp_path / "features.csv"
    out = add_calendar_features(input_file, output_file)

    assert output_file.exists()
    assert "day_of_week" in out.columns
    assert out["is_weekend"].iloc[0] == 1


def test_add_lag_features(tmp_path: Path) -> None:
    input_file = tmp_path / "features.csv"
    df = _make_hourly_df("2024-01-01 00:00:00", 200)
    df["hour"] = df["HR"]
    df.to_csv(input_file, index=False)

    output_file = tmp_path / "lags.csv"
    out = add_lag_features(input_file, output_file)

    assert output_file.exists()
    assert "load_lag_24" in out.columns
    assert len(out) == 32


def test_train_load_forecaster_stubbed(tmp_path: Path, monkeypatch) -> None:
    df = _make_hourly_df("2024-01-01 00:00:00", 300)
    df["hour"] = df["HR"]
    df["day_of_week"] = 1
    df["month"] = 1
    df["is_weekend"] = 0
    df["is_holiday"] = 0
    df["is_peak_hour"] = 0
    df["load_lag_24"] = df["CAISO"].shift(24)
    df["load_lag_168"] = df["CAISO"].shift(168)
    df["load_rolling_mean_24"] = df["CAISO"].shift(24).rolling(24).mean()
    df = df.dropna()

    input_file = tmp_path / "ready.csv"
    df.to_csv(input_file, index=False)

    class StubModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [float(X["hour"].mean())] * len(X)

    import load_forecasting_cali.model as model_mod

    monkeypatch.setattr(model_mod.xgb, "XGBRegressor", lambda **_: StubModel())
    plot_file = tmp_path / "plot.png"
    metrics = train_load_forecaster(input_file, plot_file)

    assert plot_file.exists()
    assert "mae" in metrics
    assert "mape" in metrics


def test_add_weather_features_stubbed(tmp_path: Path, monkeypatch) -> None:
    df = _make_hourly_df("2024-01-01 00:00:00", 48)
    df["hour"] = df["HR"]
    input_file = tmp_path / "lags.csv"
    df.to_csv(input_file, index=False)

    import load_forecasting_cali.weather as weather_mod

    class StubHourly:
        def __init__(self, _coords, start, end):
            self.start = start
            self.end = end

        def fetch(self):
            idx = pd.date_range(self.start, self.end, freq="h")
            out = pd.DataFrame({"temp": [20.0] * len(idx)}, index=idx)
            out.index.name = "time"
            return out

    monkeypatch.setattr(weather_mod, "hourly", lambda coords, start, end: StubHourly(coords, start, end))

    output_file = tmp_path / "final.csv"
    out = weather_mod.add_weather_features(input_file, output_file)

    assert output_file.exists()
    assert "temp_la" in out.columns
    assert "temp_sf" in out.columns
    assert "la_cdh" in out.columns
