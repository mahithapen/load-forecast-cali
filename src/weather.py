from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from meteostat import Point, hourly

# Meteostat blocks a single hourly() call spanning more than ~3 years.
_MAX_HOURLY_CHUNK = timedelta(days=700)


def _fetch_hourly_temperature(coords: Point, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch hourly temperature, chunking to stay under Meteostat's large-request limit."""
    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + _MAX_HOURLY_CHUNK)
        data = hourly(coords, chunk_start, chunk_end)
        part = data.fetch()
        if part is not None and not part.empty and "temp" in part.columns:
            frames.append(part[["temp"]])
        if chunk_end >= end:
            break
        chunk_start = chunk_end + timedelta(hours=1)

    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


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
        weather_df = _fetch_hourly_temperature(coords, start, end)
        if weather_df is not None and not weather_df.empty:
            weather_df = weather_df.rename(columns={"temp": col_name})
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
