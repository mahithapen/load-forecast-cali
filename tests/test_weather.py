from unittest.mock import MagicMock, patch
import pandas as pd
from load_forecasting_cali.weather import add_weather_features


@patch("load_forecasting_cali.weather.Hourly")
def test_add_weather_features(mock_hourly, tmp_path):
    # Mock Meteostat response
    mock_data = MagicMock()
    mock_df = pd.DataFrame(
        {"temp": [20.0]}, index=pd.to_datetime(["2023-01-01 00:00:00"]))
    mock_data.fetch.return_value = mock_df
    mock_hourly.return_value = mock_data

    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame({"DATE": ["2023-01-01"], "hour": [0]}
                 ).to_csv(input_csv, index=False)

    add_weather_features(input_csv, output_csv)
    result = pd.read_csv(output_csv)
    assert "temp_la" in result.columns
    assert "la_cdh" in result.columns  # Cooling Degree Hours
