"""
Testing on NIR artificially generated data
Developed by Md Mahmudul Haque
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def test_model(model: object, X: np.ndarray, config: dict) -> np.float64:
    """testing model to output one score

    Args:
        model (object): pls regression model
        X (np.ndarray): numpy n-dim array for testing
        config (dict): config for smoothing according to train data

    Returns:
        np.float64: mean prediction
    """
    # smoothing
    X_smoothed = savgol_filter(
        X[:, config["MIN_WL"]:config["MAX_WL"]],
        window_length=config["WINDOW"],
        polyorder = config["POLYNOMIAL"],
        deriv=config["DOF"]
    )

    # predicting
    y_out = model.predict(X_smoothed)

    # averaging the predictions
    y_mean = y_out.mean()

    return y_mean

if __name__=="__main__":
    """the test data is based on one machine's multiple scan
    """
    working_dir = os.getcwd()
    parent_dir = "/".join(working_dir.split("/")[:-1])
    model_dir = f"{parent_dir}/model"
    data_dir = f"{parent_dir}/data"
    data_list = os.listdir(data_dir)

    with open(f"{working_dir}/config.json", "r") as file:
        config = json.load(file)

    print(f"[Info:] Parent Directory: {parent_dir}")
    print(f"[Info:] Working Directory: {working_dir}")
    print(f"[Info:] Data Directory: {data_list}")
    print(f"[Info:] Model Directory: {model_dir}")

    df = pd.read_csv(f"{data_dir}/test_df.csv")
    with open(f"{model_dir}/pls_regression_n_comp_33.pkl", "rb") as file:
        model = pickle.load(file)

    X = df.values
    score = test_model(model, X, config)
    print(f"[Info:] Score: {score}")
    print(f"[Info:] Type: {type(score)}")
