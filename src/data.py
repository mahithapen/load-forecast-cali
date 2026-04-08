from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd


def merge_caiso_data(input_dir: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Merge CAISO hourly load Excel files into a single CSV."""
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    all_files = glob.glob(str(input_dir / "*.xlsx"))

    if not all_files:
        raise FileNotFoundError(f"No Excel files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for filename in all_files:
        df = pd.read_excel(filename, index_col=None, header=0)
        df.columns = [str(c).strip().upper() for c in df.columns]
        if "DATE" in df.columns and "HR" in df.columns:
            frames.append(df)
        else:
            continue

    if not frames:
        raise ValueError("No valid CAISO sheets found with DATE and HR columns.")

    full_df = pd.concat(frames, axis=0, ignore_index=True)
    full_df = full_df.sort_values(by=["DATE", "HR"])
    full_df.to_csv(output_file, index=False)
    return full_df
