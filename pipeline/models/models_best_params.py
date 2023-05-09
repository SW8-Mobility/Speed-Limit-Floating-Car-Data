from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from xgboost import XGBClassifier  # type: ignore
import pandas as pd  # type: ignore
from sklearn.pipeline import make_pipeline

CORE_NUM = 15  # how many cores to use in grid_search
RANDOM_STATE = 42
# Best params are currently from Nesheim's run with no array features
MLP_BEST_PARAMS = {
    "mlpclassifier__max_iter": 1000,
    "mlpclassifier__random_state": RANDOM_STATE,
    "mlpclassifier__verbose": 3,
    "mlpclassifier__activation": "logistic",
    "mlpclassifier__alpha": 0.01,
    "mlpclassifier__hidden_layer_sizes": (50, 50),
    "mlpclassifier__learning_rate": "constant",
    "mlpclassifier__solver": "adam",
}
RF_BEST_PARAMS = {
    "random_state": RANDOM_STATE,
    "verbose": 3,
    "max_depth": 10,
    "max_features": "auto",
    "min_samples_leaf": 4,
    "min_samples_split": 2,
    "n_estimators": 200,
}
XGBOOST_BEST_PARAMS = {
    "random_state": RANDOM_STATE,
    "verbosity": 2,
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 100,
    "subsample": 0.8,
}
LG_BEST_PARAMS = {
    "random_state": RANDOM_STATE,
    "verbose": 3,
    "C": 1,
    "max_iter": 1000,
    "penalty": "l2",
    "solver": "newton-cg",
    "tol": 0.0001,
}


def create_mlp_best_params(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Create MLP with the best parameters
    Args:
        x_train: a training set
        y_train: a set of targets for training set

    Returns: an MLP classifier (Pipeline)

    """

    # Create pipeline with StandardScaler and MLPClassifier
    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(**MLP_BEST_PARAMS),
    )

    pipeline.fit(x_train, y_train)

    return pipeline


def create_rf_best_params(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Create RF with best parameters
    Args:
        x_train: a training set
        y_train: a set of targets for training set

    Returns: a RandomForestRegressor

    """
    rfr = RandomForestRegressor(**RF_BEST_PARAMS)

    rfr.fit(x_train, y_train)

    return rfr


def create_xgboost_best_params(x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Create XGBoost with best parameters
    Args:
        x_train: a training set
        y_train: a set of targets for training set

    Returns: an XGBClassifier

    """

    xgb = XGBClassifier(**RF_BEST_PARAMS)

    xgb.fit(x_train, y_train)

    return xgb


def create_logistic_regression_best_params(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Create logistical regression with best parameters
    Args:
        x_train: a training set
        y_train: a set of targets for training set

    Returns: a LogisticRegression

    """
    lg = LogisticRegression(**RF_BEST_PARAMS)

    lg.fit(x_train, y_train)

    return lg

def main():
    # https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
    # importing required libraries
    # importing Scikit-learn library and datasets package
    from sklearn import datasets

    # Loading the iris plants dataset (classification)
    iris = datasets.load_iris()

    create_rf_best_params()

if __name__ == '__main__':
    main()