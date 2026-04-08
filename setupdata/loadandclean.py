from load_forecasting_cali.data import merge_caiso_data


if __name__ == "__main__":
    merge_caiso_data("caiso_load_data", "caiso_load_complete.csv")
