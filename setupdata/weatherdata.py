from load_forecasting_cali.weather import add_weather_features


if __name__ == "__main__":
    add_weather_features("caiso_model_ready.csv", "caiso_final_dataset.csv")
