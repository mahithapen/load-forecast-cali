from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
from meteostat import Point, hourly


def add_weather_features(input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Add weather features using Meteostat for LA and SF."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    df = pd.read_csv(input_file)

    temp_dt = pd.to_datetime(df["DATE"])
    min_dt = temp_dt.min()
    max_dt = temp_dt.max()

    start = datetime(min_dt.year, min_dt.month, min_dt.day)
    end = datetime(max_dt.year, max_dt.month, max_dt.day, 23)

    locations = {
        "temp_la": Point(34.0522, -118.2437),
        "temp_sf": Point(37.7749, -122.4194),
    }

    df["DATE_TIME"] = pd.to_datetime(df["DATE"]) + pd.to_timedelta(df["hour"], unit="h")

    for col_name, coords in locations.items():
        data = hourly(coords, start, end)
        weather_df = data.fetch()
        if weather_df is not None and not weather_df.empty:
            weather_df = weather_df[["temp"]].rename(columns={"temp": col_name})
            df = pd.merge_asof(
                df.sort_values("DATE_TIME"),
                weather_df.sort_values(weather_df.index.name or "time"),
                left_on="DATE_TIME",
                right_index=True,
                direction="nearest",
            )

    if "temp_la" in df.columns:
        df["la_cdh"] = df["temp_la"].apply(lambda x: max(0, x - 18.3))

    df.to_csv(output_file, index=False)
    return df
