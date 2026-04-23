from features import add_lag_features


if __name__ == "__main__":
    add_lag_features("caiso_features.csv", "caiso_model_ready.csv")
