from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import GridSearchCV, KFold, train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from xgboost import XGBClassifier # type: ignore
import pandas as pd # type: ignore

from pipeline.models.statistical_model import StatisticalModel

CORE_NUM = 16 # how many cores to use in grid_search
RANDOM_STATE = 42

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
    # create pipeline with mlp and scaler
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=1000, random_state=42)),
        ]
    )

    # define the hyperparameters to search over
    parameters = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__activation": ["relu", "logistic"],
        "mlp__solver": ["adam", "lbfgs"],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__early_stopping": [True, False],
        "mlp__tol": [1e-3, 1e-4, 1e-5],
        "mlp__beta_1": [0.9, 0.8, 0.7],
        "mlp__beta_2": [0.999, 0.9, 0.8],
        "mlp__validation_fraction": [0.1, 0.2, 0.3],
    }

    # perform grid search with k-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(pipeline, parameters, cv=kfold, n_jobs=CORE_NUM)
    grid_search.fit(x_train, y_train)

    # return the best random forest model and the best hyperparameters found by the grid search
    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore

def random_forest_regressor_gridsearch(
    X_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5
) -> tuple[RandomForestRegressor, dict]:
    """
    Create and tune the hyperparameters of a RandomForest regression modelusing GridSearchCV with k-fold cross-validation.
    Remember to use quantize_results when predicting. 

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best RandomForestRegressor and the best hyperparameters.
    """

    # Create a random forest regressor
    rfr = RandomForestRegressor(random_state=42)

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
        estimator=rfr, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM
    )

    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore

def xgboost_classifier_gridsearch(x_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5) -> tuple[RandomForestRegressor, dict]:
    """
    Create and tune the hyperparameters of an xgboost classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best xgboost classifier and the best hyperparameters.
    """
    # Create an XGBoost classifier
    xgb = XGBClassifier(random_state=RANDOM_STATE)

    # Set up the parameter grid to search over
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.5],
        'subsample': [0.5, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0]
    }

    # Create the grid search object
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM)

    # Fit the grid search object to the training data
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_ # type: ignore

def logistic_regression_gridsearch(X_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5) -> tuple[RandomForestRegressor, dict]:
    """
    Create and tune the hyperparameters of a Logistic regression classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best Logistic Regression Classifier and the best hyperparameters.
    """
    # Create a Logistic Regression model
    logreg = LogisticRegression(random_state=RANDOM_STATE)

    # Set up the parameter grid to search over
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['saga', 'liblinear']
    }

    # Create the grid search object
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM)

    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_ # type: ignore

def statistical_model() -> StatisticalModel:
    """Create a basic statistical model

    Returns:
        StatisticalModel: a statistical model with a predict function
    """
    return StatisticalModel()

if __name__ == "__main__":
    pass
