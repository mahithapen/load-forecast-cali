from __future__ import annotations

from pathlib import Path
import pandas as pd

from load_forecasting_cali import cli


def test_cli_merge(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    df = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01"],
            "HR": [0, 1],
            "CAISO": [1000, 1100],
        }
    )
    df.to_excel(input_dir / "a.xlsx", index=False)

    output_file = tmp_path / "merged.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "load-forecast",
            "merge",
            "--input-dir",
            str(input_dir),
            "--output-file",
            str(output_file),
        ],
    )
    assert cli.main() == 0
    assert output_file.exists()


def test_cli_pipeline(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01"],
            "HR": [0, 1],
            "CAISO": [1000, 1100],
        }
    )
    df.to_excel(raw_dir / "a.xlsx", index=False)

    merged = tmp_path / "merged.csv"
    features = tmp_path / "features.csv"
    lags = tmp_path / "lags.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "load-forecast",
            "pipeline",
            "--input-dir",
            str(raw_dir),
            "--merged-file",
            str(merged),
            "--features-file",
            str(features),
            "--lags-file",
            str(lags),
        ],
    )
    assert cli.main() == 0
    assert merged.exists()
    assert features.exists()
    assert lags.exists()
