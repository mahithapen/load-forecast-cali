from __future__ import annotations

import argparse
from pathlib import Path

from .data import merge_caiso_data
from .features import add_calendar_features, add_lag_features
from .weather import add_weather_features
from .model import train_load_forecaster, tune_load_forecaster


def _default_paths():
    return {
        "raw_dir": Path("data/raw/caiso_load_data"),
        "merged": Path("data/processed/caiso_load_complete.csv"),
        "features": Path("data/processed/caiso_features.csv"),
        "lags": Path("data/processed/caiso_model_ready.csv"),
        "final": Path("data/processed/caiso_final_dataset.csv"),
        "plot": Path("artifacts/forecast_check.png"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAISO load forecasting pipeline CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths = _default_paths()

    merge = subparsers.add_parser("merge", help="Merge raw CAISO Excel files.")
    merge.add_argument("--input-dir", default=paths["raw_dir"])
    merge.add_argument("--output-file", default=paths["merged"])

    cal = subparsers.add_parser("calendar", help="Add calendar features.")
    cal.add_argument("--input-file", default=paths["merged"])
    cal.add_argument("--output-file", default=paths["features"])

    lags = subparsers.add_parser("lags", help="Add lag/rolling features.")
    lags.add_argument("--input-file", default=paths["features"])
    lags.add_argument("--output-file", default=paths["lags"])

    weather = subparsers.add_parser("weather", help="Add weather features.")
    weather.add_argument("--input-file", default=paths["lags"])
    weather.add_argument("--output-file", default=paths["final"])

    train = subparsers.add_parser("train", help="Train the XGBoost model.")
    train.add_argument("--input-file", default=paths["lags"])
    train.add_argument("--plot-file", default=paths["plot"])
    train.add_argument(
        "--validation",
        choices=["holdout-ratio", "holdout-months", "time-series-cv"],
        default="holdout-ratio",
        help="holdout-ratio: last fraction of time as test; holdout-months: last N calendar months; "
        "time-series-cv: expanding-window TimeSeriesSplit (mean/std metrics).",
    )
    train.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test fraction for holdout-ratio (default 0.2).",
    )
    train.add_argument(
        "--test-months",
        type=int,
        default=6,
        help="Number of calendar months in the test set for holdout-months (default 6).",
    )
    train.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of folds for time-series-cv (default 5).",
    )

    tune = subparsers.add_parser(
        "tune",
        help="Small grid search over XGBoost hyperparameters (time-series CV).",
    )
    tune.add_argument("--input-file", default=paths["lags"])
    tune.add_argument("--cv-splits", type=int, default=5)
    tune.add_argument("--n-estimators", default="100,200,400")
    tune.add_argument("--learning-rate", default="0.05,0.1")
    tune.add_argument("--max-depth", default="4,6,8")

    pipe = subparsers.add_parser(
        "pipeline", help="Run merge -> calendar -> lags.")
    pipe.add_argument("--input-dir", default=paths["raw_dir"])
    pipe.add_argument("--merged-file", default=paths["merged"])
    pipe.add_argument("--features-file", default=paths["features"])
    pipe.add_argument("--lags-file", default=paths["lags"])

    return parser


def _print_train_metrics(metrics: dict) -> None:
    print(f"MAE:  {metrics['mae']:.2f} MW")
    print(f"MAPE: {metrics['mape']:.2%}")
    if metrics.get("validation") == "time_series_cv":
        print(f"MAE std (across folds):  {metrics['mae_std']:.2f} MW")
        print(f"MAPE std (across folds): {metrics['mape_std']:.2%}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "merge":
        merge_caiso_data(args.input_dir, args.output_file)
    elif args.command == "calendar":
        add_calendar_features(args.input_file, args.output_file)
    elif args.command == "lags":
        add_lag_features(args.input_file, args.output_file)
    elif args.command == "weather":
        add_weather_features(args.input_file, args.output_file)
    elif args.command == "train":
        validation_map = {
            "holdout-ratio": "holdout_ratio",
            "holdout-months": "holdout_last_months",
            "time-series-cv": "time_series_cv",
        }
        metrics = train_load_forecaster(
            args.input_file,
            args.plot_file,
            validation=validation_map[args.validation],
            test_ratio=args.test_ratio,
            test_months=args.test_months,
            time_series_cv_splits=args.cv_splits,
        )
        _print_train_metrics(metrics)
    elif args.command == "tune":
        n_estimators_grid = [int(x) for x in str(args.n_estimators).split(",") if x.strip()]
        learning_rate_grid = [float(x) for x in str(args.learning_rate).split(",") if x.strip()]
        max_depth_grid = [int(x) for x in str(args.max_depth).split(",") if x.strip()]

        out = tune_load_forecaster(
            args.input_file,
            time_series_cv_splits=args.cv_splits,
            n_estimators_grid=n_estimators_grid,
            learning_rate_grid=learning_rate_grid,
            max_depth_grid=max_depth_grid,
        )
        best = out["best"]
        p = best["params"]
        print("Best params (by mean MAE across folds):")
        print(
            f"  n_estimators={p['n_estimators']}, learning_rate={p['learning_rate']}, "
            f"max_depth={p['max_depth']}"
        )
        print(
            f"CV MAE:  {best['mae']:.2f} MW (std {best['mae_std']:.2f})"
        )
        print(
            f"CV MAPE: {best['mape']:.2%} (std {best['mape_std']:.2%})"
        )
    elif args.command == "pipeline":
        merge_caiso_data(args.input_dir, args.merged_file)
        add_calendar_features(args.merged_file, args.features_file)
        add_lag_features(args.features_file, args.lags_file)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
