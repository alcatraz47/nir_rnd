"""
Training script on NIR artificially generated data
Developed by Md Mahmudul Haque
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import f
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


def remove_outliers(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """removes outliers from the dataset using Q-residuals and Hotelling's T-squared method
    based on Partial Least Squares (PLS) regression.

    firstly, fits a PLS regression model with a specified number of components;
    then computes the score matrix (T), loading matrix (P), and error matrix (E);
    after that calculates Q-residuals and T-squared statistics;
    then determines the confidence intervals for Q-residuals and T-squared statistics;
    identifies outliers based on these confidence intervals;
    lastly, removes outliers from the predictor and response matrices.

    Args:
        X (np.ndarray): NIR spectroscopy data
        y (np.ndarray): response variable

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: predictor and response matrices with outliers removed
    """
    n_components = 10
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    # to find out score matrix T is being calculated
    # then loading matrix from pls
    # after that error matrix is being calculated
    # two scatter technique is used so the equation
    # is X = T (dot) P.T + E
    T_matrix = pls.x_scores_
    P_matrix = pls.x_loadings_
    E_matrix = X - np.dot(T_matrix, P_matrix.T)

    # now to derive Q-residuals
    Q_res = np.sum(E_matrix ** 2, axis=1)

    # to calculate the useful data points T-squared needed
    T_matrix_squared = np.sum((T_matrix / np.std(T_matrix, axis=0))**2, axis=1)

    # calculating from T_matrix_squared to Hotelling's T-squared
    # which is distributed according to F-distribution
    # a library is getting used as mathematical explanation for me
    # is kind of not cleared here

    conf_interval = 0.95

    T_matrix_squared_conf =  f.ppf(
                                q=conf_interval,
                                dfn=n_components,
                                dfd=(X.shape[0]-n_components)
                            )*n_components*(X.shape[0]-1)/(X.shape[0]-n_components)

    # estimation of the confidence level for the Q-res
    i = np.max(Q_res)+1
    while 1-np.sum(Q_res>i)/np.sum(Q_res>0)> conf_interval:
        i -= 1
    Q_conf = i

    # extracting outlier indices
    outliers = ((Q_res > Q_conf) | (T_matrix_squared > T_matrix_squared_conf))

    # removing outliers
    X_no_outlier = X[~outliers]
    y_no_outlier = Y[~outliers]

    return X_no_outlier, y_no_outlier

def select_feature_plsr_cv(X: np.ndarray, y: np.ndarray, n_components: int, cv: int=5):
    """selects feature that are minimum required for the PLSRegression
    to learn properly.

    selects 1 to n_components; creates PLSRegression model;
    then uses cross validation to train and predict. lastly,
    calculates mse, and with minimum mse; selects the minimum
    required components as features and trains the model to
    calculate further statistical result analysis (r^2 and mse)
    with cross validation and single calibration

    Args:
        X (np.ndarray): numpy array of input feature without outliers
        y (np.ndarray): numpy array of ground truth without outliers
        n_components (int): numper of maximum components
        cv (int, optional): integer cross validation. Defaults to 5.
    Returns:
        (int): minimum components needed to train
    """
    mse = []
    components = np.arange(1, n_components)

    for component in components:
        plsr = PLSRegression(n_components=component)
        y_cross_val = cross_val_predict(plsr, X, y, cv=cv)
        mse.append(mean_squared_error(y, y_cross_val))
        print(f"[Info:] Last Trained on Number of Components: {component}")
        print(f"[Info:] MSE: {mse[-1]}")

    msemin = np.argmin(mse)
    print(f"[Info:] Minimum no. of Components Needed: {msemin+1}")
    return msemin+1


def train_plsr_model(X: np.ndarray, y: np.ndarray, n_components: int, cross_val: int = 5)-> object:
    """
    trains a Partial Least Squares Regression (PLSR) model and evaluates its performance
    using calibration and cross-validation metrics.

    fistly, trains a PLSR model with the specified number of components;
    predicts the response variable using the calibrated model;
    performs cross-validation prediction;
    calculates R-squared and Mean Squared Error (MSE) for both calibration and cross-validation;
    prints the R-squared and MSE values for calibration and cross-validation.

    Args:
        X (np.ndarray): predictor matrix (NIR spectroscopy data).
        y (np.ndarray): response variable.
        n_components (int): number of components to use in the PLSR model.
        cross_val (int, optional): number of folds for cross-validation. Defaults to 5.

    Returns:
        object: The trained PLSRegression model.
    """
    plsr_selected = PLSRegression(n_components=n_components)
    plsr_selected.fit(X, y)
    y_pred_calibrated = plsr_selected.predict(X)
    y_cross_val_selected = cross_val_predict(plsr_selected, X, y, cv=cross_val)

    r2_score_calibrated = r2_score(y, y_pred_calibrated)
    r2_score_cv = r2_score(y, y_cross_val_selected)

    mse_calibrated = mean_squared_error(y, y_pred_calibrated)
    mse_cv = mean_squared_error(y, y_cross_val_selected)

    print(f"[Info:] R2 Calibrated: {r2_score_calibrated:.4f}")
    print(f"[Info:] R2 CV: {r2_score_cv:.4f}")
    print(f"[Info:] MSE Calibrated: {mse_calibrated:.4f}")
    print(f"[Info:] MSE CV: {mse_cv:.4f}")

    return plsr_selected

if __name__ == "__main__":
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

    df = pd.read_excel(f"{data_dir}/final_data.xlsx")

    X = df.values[:, 1:]
    Y = df.values[:, 0]

    # smoothing
    X_smoothed = savgol_filter(
        X[:, config["MIN_WL"]:config["MAX_WL"]],
        window_length=config["WINDOW"],
        polyorder=config["POLYNOMIAL"],
        deriv=config["DOF"]
    )
    # removing outliers
    X_no_outlier, y_no_outlier = remove_outliers(X_smoothed, Y)

    print("[Info:] Before Outlier Removal Dimensions of X and Y are:")
    print(X_smoothed.shape)
    print(Y.shape)

    print("[Info:] After Outlier Removal Dimensions of X and Y are:")
    print(X_no_outlier.shape)
    print(y_no_outlier.shape)

    # selecting components
    total_component = select_feature_plsr_cv(X_no_outlier, y_no_outlier, X_no_outlier.shape[1] // 2, 5)

    # training the model
    plsr = train_plsr_model(X_no_outlier, y_no_outlier, total_component, cross_val=5)

    # NOTE: personally I do not want to write model using pickle or joblib
    # the version problem is too much to handle, I would rather export into
    # ONNX for serving and for re-purposing ONNX to sklearn again but for
    # simplicity using joblib based export below
    model_file_name = f"pls_regression_n_comp_{total_component}"
    with open(f"{model_dir}/{model_file_name}.pkl", "wb") as file:
        pickle.dump(plsr, file)

    with open(f"{model_dir}/{model_file_name}_component.json", "w") as f:
        config_dict = {
            "MIN_COMPONENT": int(total_component)
        }
        f.write(json.dumps(config_dict))