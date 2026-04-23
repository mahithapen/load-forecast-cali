import pandas as pd
import pytest
from pathlib import Path
from load_forecasting_cali.data import merge_caiso_data


def test_merge_caiso_data(tmp_path):
    # Create a dummy Excel file
    df = pd.DataFrame({"DATE": ["2023-01-01"], "HR": [1], "CAISO": [100]})
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    df.to_excel(input_dir / "test.xlsx", index=False)

    output_file = tmp_path / "merged.csv"
    merge_caiso_data(input_dir, output_file)

    assert output_file.exists()
    result = pd.read_csv(output_file)
    assert "DATE" in result.columns
    assert result["CAISO"].iloc[0] == 100
