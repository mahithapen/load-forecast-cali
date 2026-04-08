from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb


DEFAULT_FEATURES: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_peak_hour",
    "load_lag_24",
    "load_lag_168",
    "load_rolling_mean_24",
]


def train_load_forecaster(
    input_file: str | Path,
    output_plot: Optional[str | Path] = "forecast_check.png",
    features: Optional[Iterable[str]] = None,
) -> dict:
    """Train an XGBoost model and return evaluation metrics."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    df = pd.read_csv(input_file)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["DATE_TIME"] = df["DATE"] + pd.to_timedelta(df["hour"], unit="h")

    feature_list = list(features) if features is not None else DEFAULT_FEATURES
    target = "CAISO"

    df = df.dropna(subset=feature_list + [target])
    df = df.sort_values("DATE_TIME")
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if len(test_df) == 0:
        raise ValueError("Not enough data to create a test split.")

    X_train, y_train = train_df[feature_list], train_df[target]
    X_test, y_test = test_df[feature_list], test_df[target]

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    if output_plot is not None:
        output_plot = Path(output_plot)
        plt.figure(figsize=(12, 6))
        plt.plot(test_df["DATE_TIME"].iloc[:168], y_test.iloc[:168], label="Actual")
        plt.plot(
            test_df["DATE_TIME"].iloc[:168],
            preds[:168],
            label="Predicted",
            linestyle="--",
        )
        plt.legend()
        plt.title("CAISO Load Forecast (Next 7 Days)")
        plt.tight_layout()
        plt.savefig(output_plot)

    return {"mae": float(mae), "mape": float(mape)}
