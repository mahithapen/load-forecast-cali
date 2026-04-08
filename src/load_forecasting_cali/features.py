from __future__ import annotations

from pathlib import Path
import pandas as pd
import holidays


def add_calendar_features(input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Add calendar-based features to the merged CAISO dataset."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    df = pd.read_csv(input_file)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    df["hour"] = pd.to_numeric(df["HR"], errors="coerce")
    df = df.dropna(subset=["hour"])

    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["month"] = df["DATE"].dt.month
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    us_holidays = holidays.US()
    df["is_holiday"] = df["DATE"].apply(lambda x: 1 if x in us_holidays else 0)

    df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if 16 <= x <= 21 else 0)

    df.to_csv(output_file, index=False)
    return df


def add_lag_features(input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Add lag and rolling mean features to the dataset."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    df = pd.read_csv(input_file)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(by=["DATE", "hour"])

    df["CAISO"] = pd.to_numeric(df["CAISO"], errors="coerce")
    df["CAISO"] = df["CAISO"].interpolate(method="linear")

    df["load_lag_24"] = df["CAISO"].shift(24)
    df["load_lag_168"] = df["CAISO"].shift(168)
    df["load_rolling_mean_24"] = df["CAISO"].shift(24).rolling(window=24).mean()

    df_clean = df.dropna(subset=["load_lag_168"])
    df_clean.to_csv(output_file, index=False)
    return df_clean
