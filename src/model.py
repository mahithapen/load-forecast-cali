from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
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

ValidationMode = Literal["holdout_ratio", "holdout_last_months", "time_series_cv"]


def _load_and_prepare(
    input_file: str | Path,
    feature_list: list[str],
    *,
    target: str = "CAISO",
) -> pd.DataFrame:
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    df = pd.read_csv(input_file)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["DATE_TIME"] = df["DATE"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.dropna(subset=feature_list + [target])
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Not enough rows to train after dropping NaNs.")
    return df


def tune_load_forecaster(
    input_file: str | Path,
    *,
    features: Optional[Iterable[str]] = None,
    time_series_cv_splits: int = 5,
    n_estimators_grid: Iterable[int] = (100, 200, 400),
    learning_rate_grid: Iterable[float] = (0.05, 0.1),
    max_depth_grid: Iterable[int] = (4, 6, 8),
    random_state: int = 0,
) -> dict:
    """Small deterministic grid search using expanding-window time-series CV.

    Selects hyperparameters by **minimizing mean MAE** across folds.
    """
    feature_list = list(features) if features is not None else DEFAULT_FEATURES
    df = _load_and_prepare(input_file, feature_list)
    target = "CAISO"

    if time_series_cv_splits < 2:
        raise ValueError("time_series_cv_splits must be at least 2.")

    X = df[feature_list].to_numpy()
    y = df[target].to_numpy()
    tsc = TimeSeriesSplit(n_splits=time_series_cv_splits)

    results: list[dict] = []

    for n_estimators, learning_rate, max_depth in product(
        list(n_estimators_grid),
        list(learning_rate_grid),
        list(max_depth_grid),
    ):
        fold_mae: list[float] = []
        fold_mape: list[float] = []

        for train_idx, test_idx in tsc.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = xgb.XGBRegressor(
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=int(max_depth),
                objective="reg:squarederror",
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_mae.append(float(mean_absolute_error(y_test, preds)))
            fold_mape.append(float(mean_absolute_percentage_error(y_test, preds)))

        results.append(
            {
                "params": {
                    "n_estimators": int(n_estimators),
                    "learning_rate": float(learning_rate),
                    "max_depth": int(max_depth),
                    "random_state": int(random_state),
                    "objective": "reg:squarederror",
                },
                "mae": float(np.mean(fold_mae)),
                "mape": float(np.mean(fold_mape)),
                "mae_std": float(np.std(fold_mae)),
                "mape_std": float(np.std(fold_mape)),
            }
        )

    best = sorted(results, key=lambda r: (r["mae"], r["mape"]))[0]
    return {
        "validation": "time_series_cv_grid_search",
        "time_series_cv_splits": int(time_series_cv_splits),
        "best": best,
        "results": results,
    }


def _split_holdout_ratio(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive).")
    split_idx = int(len(df) * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def _split_holdout_last_months(df: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_months < 1:
        raise ValueError("test_months must be at least 1.")
    end = df["DATE_TIME"].max()
    cutoff = end - pd.DateOffset(months=test_months)
    test_df = df[df["DATE_TIME"] >= cutoff]
    train_df = df[df["DATE_TIME"] < cutoff]
    return train_df, test_df


def train_load_forecaster(
    input_file: str | Path,
    output_plot: Optional[str | Path] = "forecast_check.png",
    features: Optional[Iterable[str]] = None,
    *,
    validation: ValidationMode = "holdout_ratio",
    test_ratio: float = 0.2,
    test_months: int = 6,
    time_series_cv_splits: int = 5,
) -> dict:
    """Train an XGBoost model and return evaluation metrics.

    * ``holdout_ratio`` — chronological split: last ``test_ratio`` of rows are test (default, same as before).
    * ``holdout_last_months`` — test set is all rows from the last ``test_months`` calendar months.
    * ``time_series_cv`` — expanding-window CV;
      reports mean and std of MAE/MAPE across folds (and per-fold lists). The plot shows the last fold's test segment.
    """
    feature_list = list(features) if features is not None else DEFAULT_FEATURES
    target = "CAISO"
    df = _load_and_prepare(input_file, feature_list, target=target)

    if validation == "time_series_cv":
        return _train_time_series_cv(
            df,
            feature_list,
            target,
            output_plot,
            n_splits=time_series_cv_splits,
        )

    if validation == "holdout_ratio":
        train_df, test_df = _split_holdout_ratio(df, test_ratio)
    elif validation == "holdout_last_months":
        train_df, test_df = _split_holdout_last_months(df, test_months)
    else:
        raise ValueError(f"Unknown validation mode: {validation!r}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Train or test split is empty; use more data, a smaller test_ratio, "
            "or fewer test_months."
        )

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
        _save_holdout_plot(
            Path(output_plot),
            test_df["DATE_TIME"].iloc[:168],
            y_test.iloc[:168],
            preds[:168],
        )

    out = {
        "mae": float(mae),
        "mape": float(mape),
        "validation": validation,
    }
    if validation == "holdout_ratio":
        out["test_ratio"] = test_ratio
    else:
        out["test_months"] = test_months
    return out


def _train_time_series_cv(
    df: pd.DataFrame,
    feature_list: list[str],
    target: str,
    output_plot: Optional[str | Path],
    *,
    n_splits: int,
) -> dict:
    if n_splits < 2:
        raise ValueError("time_series_cv_splits must be at least 2.")
    n_samples = len(df)
    if n_samples < n_splits + 2:
        raise ValueError(
            f"Need at least n_splits + 2 samples for time-series CV; got {n_samples} rows, "
            f"n_splits={n_splits}."
        )

    X = df[feature_list].to_numpy()
    y = df[target].to_numpy()
    tsc = TimeSeriesSplit(n_splits=n_splits)

    fold_mae: list[float] = []
    fold_mape: list[float] = []
    last_preds: np.ndarray | None = None
    last_y_test: np.ndarray | None = None
    last_date_time: pd.Series | None = None

    for train_idx, test_idx in tsc.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_mae.append(float(mean_absolute_error(y_test, preds)))
        fold_mape.append(float(mean_absolute_percentage_error(y_test, preds)))
        last_preds = preds
        last_y_test = y_test
        last_date_time = df["DATE_TIME"].iloc[test_idx]

    assert last_preds is not None and last_y_test is not None and last_date_time is not None

    if output_plot is not None:
        n_show = min(168, len(last_preds))
        _save_holdout_plot(
            Path(output_plot),
            last_date_time.iloc[:n_show],
            pd.Series(last_y_test[:n_show]),
            last_preds[:n_show],
            title_suffix=" (last CV fold, test segment)",
        )

    return {
        "mae": float(np.mean(fold_mae)),
        "mape": float(np.mean(fold_mape)),
        "mae_std": float(np.std(fold_mae)),
        "mape_std": float(np.std(fold_mape)),
        "fold_mae": fold_mae,
        "fold_mape": fold_mape,
        "validation": "time_series_cv",
        "time_series_cv_splits": n_splits,
    }


def _save_holdout_plot(
    output_plot: Path,
    date_times: pd.Series,
    y_actual: pd.Series,
    y_pred: np.ndarray | pd.Series,
    title_suffix: str = "",
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(date_times, y_actual, label="Actual")
    plt.plot(date_times, y_pred, label="Predicted", linestyle="--")
    plt.legend()
    plt.title("CAISO Load Forecast (Next 7 Days)" + title_suffix)
    plt.tight_layout()
    plt.savefig(output_plot)
