from load_forecasting_cali.model import train_load_forecaster


if __name__ == "__main__":
    metrics = train_load_forecaster("caiso_model_ready.csv", "forecast_check.png")
    print(f"MAE: {metrics['mae']:.2f} MW")
    print(f"MAPE: {metrics['mape']:.2%}")
