from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import GridSearchCV, KFold, train_test_split  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from xgboost import XGBClassifier  # type: ignore
import pandas as pd  # type: ignore
from sklearn.pipeline import make_pipeline

from pipeline.models.statistical_model import StatisticalModel

CORE_NUM = 15  # how many cores to use in grid_search
RANDOM_STATE = 42
MLP_BEST_PARAMS = {'mlpclassifier__max_iter': 1000, 'mlpclassifier__random_state': RANDOM_STATE, 'mlpclassifier__verbose': 3, 'mlpclassifier__activation': 'logistic', 'mlpclassifier__alpha': 0.01, 'mlpclassifier__hidden_layer_sizes': (50, 50), 'mlpclassifier__learning_rate': 'constant', 'mlpclassifier__solver': 'adam'}
RF_BEST_PARAMS = {'random_state': RANDOM_STATE, 'verbose': 3, 'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
XGBOOST_BEST_PARAMS = {'random_state': RANDOM_STATE, 'verbosity': 2, 'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}
LG_BEST_PARAMS = {'random_state': RANDOM_STATE, 'verbose': 3, 'C': 1, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'newton-cg', 'tol': 0.0001}

def create_mlp_best_params(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Create MLP with the best parameters
    Args:
        x_train: a training set
        y_train: a set of targets for training set

    Returns: an MLP classifier

    """

    # Create pipeline with StandardScaler and MLPClassifier
    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(**MLP_BEST_PARAMS),
    )

    pipeline.fit(x_train, y_train)

    return pipeline

def create_rf_best_params(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    rfr = RandomForestRegressor(**RF_BEST_PARAMS)

    rfr.fit(x_train, y_train)

    return rfr

def create_xgboost_best_params(x_train: pd.DataFrame, y_train: pd.Series):
    xgb = XGBClassifier(**RF_BEST_PARAMS)

    xgb.fit(x_train, y_train)

    return xgb

def create_logistic_regression_best_params(x_train: pd.DataFrame, y_train: pd.Series):
    lg = LogisticRegression(**RF_BEST_PARAMS)

    lg.fit(x_train, y_train)

    return lg

def create_mlp_grid_search(
    x_train: pd.DataFrame, y_train: pd.Series, k: int = 5
) -> tuple[MLPClassifier, dict]:
    """
    Create and tune the hyperparameters of an MLP classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best MLP classifier and the best hyperparameters.
    """
    param_grid = {
        "mlpclassifier__hidden_layer_sizes": [(100,), (50, 50), (20, 20, 20)],
        "mlpclassifier__activation": ["relu", "tanh", "logistic"],
        "mlpclassifier__solver": ["sgd", "adam"],
        "mlpclassifier__alpha": [0.0001, 0.001, 0.01],
        "mlpclassifier__learning_rate": ["constant", "adaptive"],
    }

    # Create pipeline with StandardScaler and MLPClassifier
    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE, verbose=3),
    )

    # perform grid search with k-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM, verbose=3
    )
    grid_search.fit(x_train, y_train)

    # return the best mlp model and the best hyperparameters found by the grid search
    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore


def random_forest_regressor_gridsearch(
    x_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5
) -> tuple[RandomForestRegressor, dict]:
    """
    Create and tune the hyperparameters of a RandomForest regression model using GridSearchCV with k-fold cross-validation.
    Remember to use quantize_results when predicting.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best RandomForestRegressor and the best hyperparameters.
    """

    # Create a random forest regressor
    rfr = RandomForestRegressor(random_state=42, verbose=3)

    # Set up the parameter grid to search over
    param_grid = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [5, 10, 20, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    }

    # Create the grid search object
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rfr, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM, verbose=3
    )

    # Fit the grid search object to the training data
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore


def xgboost_classifier_gridsearch(
    x_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5
) -> tuple[RandomForestRegressor, dict]:
    """
    Create and tune the hyperparameters of an xgboost classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best xgboost classifier and the best hyperparameters.
    """
    # Create an XGBoost classifier
    xgb = XGBClassifier(random_state=RANDOM_STATE, verbosity=2)

    # Set up the parameter grid to search over
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.5],
        "subsample": [0.5, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
    }

    # Create the grid search object
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=xgb, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM, verbose=3
    )

    # Fit the grid search object to the training data
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore


def logistic_regression_gridsearch(
    x_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5
) -> tuple[RandomForestRegressor, dict]:
    """
    Create and tune the hyperparameters of a Logistic regression classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best Logistic Regression Classifier and the best hyperparameters.
    """
    # Create a Logistic Regression model
    logreg = LogisticRegression(random_state=RANDOM_STATE, verbose=3)

    # Set up the parameter grid to search over
    parameters = {
        "penalty": ["l2"],
        "tol": [1e-4, 1e-5, 1e-6],
        "C": range(1, 11, 3),
        "solver": ["sag", "newton-cg", "lbfgs"],
        "max_iter": [1000],
    }

    # Create the grid search object
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=parameters,
        cv=kfold,
        n_jobs=CORE_NUM,
        error_score="raise",
        verbose=3,
    )

    # Fit the grid search object to the training data
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore


def statistical_model(
    x_train: pd.DataFrame, y_train: pd.DataFrame
) -> tuple[StatisticalModel, dict]:
    """Create a basic statistical model

    Returns:
        StatisticalModel: a statistical model with a predict function
    """
    return StatisticalModel(), None  # type: ignore


if __name__ == "__main__":
    pass
